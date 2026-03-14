"""Response cache — thread-safe LRU cache with TTL for Gemini API responses.

Caches non-streaming, non-tool-use ``GenerationResult`` objects to avoid
redundant API calls when users ask the same (or identical) questions.

Disabled by default — must be explicitly opted-in via ``CacheConfig.enabled``.

Cache keys are derived from: query text + model ID + system instruction.
Conversation history is NOT included in the key because identical queries
with different history contexts should produce different results.  This
means the cache is most effective for stateless/single-turn queries.

Thread-safety: all public methods acquire ``_lock`` before mutating state.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

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

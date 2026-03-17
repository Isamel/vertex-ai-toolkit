"""Tests for ToolResultCache (vaig.core.cache).

Covers:
- Basic put/get operations
- Cache miss (not cached)
- TTL expiration
- Error results never cached
- Custom TTL override via get_or_none
- LRU eviction at max_size
- Cache stats accuracy (hits, misses, evictions)
- Clear resets everything (entries + stats)
- Thread safety (concurrent access)
- Key determinism (same args different order → same key)
- Edge cases (max_size=1, default_ttl=0)
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from vaig.core.cache import ToolResultCache, _make_tool_cache_key
from vaig.tools.base import ToolResult

# ── Key generation tests ─────────────────────────────────────


class TestMakeToolCacheKey:
    """Tests for _make_tool_cache_key()."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        k1 = _make_tool_cache_key("kubectl_get", {"resource": "pods"})
        k2 = _make_tool_cache_key("kubectl_get", {"resource": "pods"})
        assert k1 == k2

    def test_different_tool_names_different_keys(self) -> None:
        """Different tool names produce different keys."""
        k1 = _make_tool_cache_key("kubectl_get", {"resource": "pods"})
        k2 = _make_tool_cache_key("kubectl_describe", {"resource": "pods"})
        assert k1 != k2

    def test_different_args_different_keys(self) -> None:
        """Different arguments produce different keys."""
        k1 = _make_tool_cache_key("kubectl_get", {"resource": "pods"})
        k2 = _make_tool_cache_key("kubectl_get", {"resource": "services"})
        assert k1 != k2

    def test_arg_order_independent(self) -> None:
        """Dict key ordering does not affect the cache key."""
        k1 = _make_tool_cache_key("tool", {"a": 1, "b": 2, "c": 3})
        k2 = _make_tool_cache_key("tool", {"c": 3, "a": 1, "b": 2})
        assert k1 == k2

    def test_key_is_sha256_hex(self) -> None:
        """Key is a valid SHA-256 hex digest (64 chars)."""
        key = _make_tool_cache_key("tool", {"x": "y"})
        assert len(key) == 64
        int(key, 16)  # valid hex

    def test_empty_args(self) -> None:
        """Empty args dict produces a valid key."""
        key = _make_tool_cache_key("tool", {})
        assert len(key) == 64

    def test_nested_args_deterministic(self) -> None:
        """Nested dicts are serialized deterministically."""
        k1 = _make_tool_cache_key("tool", {"outer": {"b": 2, "a": 1}})
        k2 = _make_tool_cache_key("tool", {"outer": {"a": 1, "b": 2}})
        assert k1 == k2


# ── ToolResultCache tests ────────────────────────────────────


class TestToolResultCache:
    """Tests for the ToolResultCache class."""

    @staticmethod
    def _ok(output: str = "ok") -> ToolResult:
        return ToolResult(output=output, error=False)

    @staticmethod
    def _err(output: str = "fail") -> ToolResult:
        return ToolResult(output=output, error=True)

    def test_put_and_get_hit(self) -> None:
        """Basic put/get cycle returns the cached value."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        result = self._ok("hello")
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, result)

        cached = cache.get(key)
        assert cached is result

    def test_cache_miss_not_cached(self) -> None:
        """Get returns None for a key that was never stored."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        assert cache.get(_make_tool_cache_key("tool", {"a": 1})) is None

    def test_cache_miss_expired_ttl(self) -> None:
        """Get returns None when the entry has expired."""
        cache = ToolResultCache(default_ttl=1, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok())

        # Immediately available
        assert cache.get(key) is not None

        # Simulate expiration
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 2):
            assert cache.get(key) is None

    def test_error_results_never_cached(self) -> None:
        """put() silently skips error results."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._err())

        assert cache.get(key) is None
        assert cache.size == 0

    def test_custom_ttl_on_put(self) -> None:
        """put() accepts a custom TTL per entry."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok(), ttl_seconds=2)

        # Still valid at 1s
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 1):
            assert cache.get(key) is not None

        # Expired at 3s
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 3):
            assert cache.get(key) is None

    def test_ttl_zero_means_no_expiration(self) -> None:
        """TTL of 0 means entries never expire."""
        cache = ToolResultCache(default_ttl=0, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok())

        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 999999):
            assert cache.get(key) is not None

    def test_lru_eviction(self) -> None:
        """LRU eviction removes the oldest entry when at capacity."""
        cache = ToolResultCache(default_ttl=60, max_size=2)

        k1 = _make_tool_cache_key("t1", {})
        k2 = _make_tool_cache_key("t2", {})
        k3 = _make_tool_cache_key("t3", {})

        cache.put(k1, self._ok("r1"))
        cache.put(k2, self._ok("r2"))
        assert cache.size == 2

        # Adding k3 evicts k1 (oldest)
        cache.put(k3, self._ok("r3"))
        assert cache.size == 2
        assert cache.get(k1) is None   # evicted
        assert cache.get(k2) is not None
        assert cache.get(k3) is not None

    def test_lru_access_refreshes_order(self) -> None:
        """Accessing an entry moves it to the end, changing eviction order."""
        cache = ToolResultCache(default_ttl=60, max_size=2)

        k1 = _make_tool_cache_key("t1", {})
        k2 = _make_tool_cache_key("t2", {})
        k3 = _make_tool_cache_key("t3", {})

        cache.put(k1, self._ok("r1"))
        cache.put(k2, self._ok("r2"))

        # Access k1 — makes it most-recently used
        cache.get(k1)

        # Adding k3 should now evict k2
        cache.put(k3, self._ok("r3"))
        assert cache.get(k1) is not None  # survived
        assert cache.get(k2) is None      # evicted
        assert cache.get(k3) is not None

    def test_update_existing_key(self) -> None:
        """Putting an existing key updates the value."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})

        r1 = self._ok("first")
        r2 = self._ok("second")

        cache.put(key, r1)
        cache.put(key, r2)

        assert cache.size == 1
        cached = cache.get(key)
        assert cached is r2

    def test_max_size_1(self) -> None:
        """Cache with max_size=1 holds at most 1 entry."""
        cache = ToolResultCache(default_ttl=60, max_size=1)
        k1 = _make_tool_cache_key("t1", {})
        k2 = _make_tool_cache_key("t2", {})

        cache.put(k1, self._ok("r1"))
        assert cache.size == 1

        cache.put(k2, self._ok("r2"))
        assert cache.size == 1
        assert cache.get(k1) is None
        assert cache.get(k2) is not None

    def test_invalid_max_size(self) -> None:
        """max_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ToolResultCache(max_size=0)

    def test_invalid_default_ttl(self) -> None:
        """Negative default_ttl raises ValueError."""
        with pytest.raises(ValueError, match="default_ttl must be >= 0"):
            ToolResultCache(default_ttl=-1)


# ── get_or_none tests ────────────────────────────────────────


class TestToolResultCacheGetOrNone:
    """Tests for get_or_none() convenience method."""

    @staticmethod
    def _ok(output: str = "ok") -> ToolResult:
        return ToolResult(output=output, error=False)

    def test_returns_cached_result(self) -> None:
        """Returns result when cached and within TTL."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok("data"))

        result = cache.get_or_none("tool", {"a": 1})
        assert result is not None
        assert result.output == "data"

    def test_returns_none_when_not_cached(self) -> None:
        """Returns None when not in cache."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        assert cache.get_or_none("tool", {"a": 1}) is None

    def test_ttl_override_shorter(self) -> None:
        """ttl_override can shorten effective TTL."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok(), ttl_seconds=60)

        # With override=2, entry should expire at 3s
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 3):
            assert cache.get_or_none("tool", {"a": 1}, ttl_override=2) is None

    def test_ttl_override_longer(self) -> None:
        """ttl_override can extend effective TTL for the lookup."""
        cache = ToolResultCache(default_ttl=2, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok(), ttl_seconds=2)

        # At 3s, stored TTL=2 would miss, but override=10 allows it
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 3):
            assert cache.get_or_none("tool", {"a": 1}, ttl_override=10) is not None

    def test_ttl_override_none_uses_stored(self) -> None:
        """When ttl_override is None, uses the stored TTL."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {"a": 1})
        cache.put(key, self._ok(), ttl_seconds=1)

        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 2):
            assert cache.get_or_none("tool", {"a": 1}) is None


# ── Stats tests ──────────────────────────────────────────────


class TestToolResultCacheStats:
    """Tests for cache statistics tracking."""

    @staticmethod
    def _ok(output: str = "ok") -> ToolResult:
        return ToolResult(output=output, error=False)

    def test_initial_stats(self) -> None:
        """Fresh cache has zero stats."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size == 0
        assert stats.max_size == 10
        assert stats.hit_rate == 0.0

    def test_miss_increments(self) -> None:
        """Cache miss increments miss counter."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        cache.get(_make_tool_cache_key("a", {}))
        cache.get(_make_tool_cache_key("b", {}))

        stats = cache.stats()
        assert stats.misses == 2
        assert stats.hits == 0

    def test_hit_increments(self) -> None:
        """Cache hit increments hit counter."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {})
        cache.put(key, self._ok())

        cache.get(key)
        cache.get(key)

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 0

    def test_hit_rate(self) -> None:
        """Hit rate is computed correctly."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {})
        cache.put(key, self._ok())

        cache.get(key)                              # hit
        cache.get(_make_tool_cache_key("miss", {})) # miss

        stats = cache.stats()
        assert stats.hit_rate == pytest.approx(0.5)

    def test_eviction_count(self) -> None:
        """Eviction counter tracks LRU evictions."""
        cache = ToolResultCache(default_ttl=60, max_size=2)

        cache.put(_make_tool_cache_key("t1", {}), self._ok())
        cache.put(_make_tool_cache_key("t2", {}), self._ok())
        cache.put(_make_tool_cache_key("t3", {}), self._ok())  # evicts t1

        stats = cache.stats()
        assert stats.evictions == 1

    def test_ttl_expiry_counts_as_eviction_and_miss(self) -> None:
        """TTL-expired entry counts as a miss and eviction."""
        cache = ToolResultCache(default_ttl=1, max_size=10)
        key = _make_tool_cache_key("tool", {})
        cache.put(key, self._ok())

        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 2):
            cache.get(key)  # expired

        stats = cache.stats()
        assert stats.misses == 1
        assert stats.evictions == 1

    def test_stats_is_frozen_snapshot(self) -> None:
        """Stats returns a frozen dataclass — not a live view."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        stats1 = cache.stats()

        cache.put(_make_tool_cache_key("tool", {}), self._ok())
        cache.get(_make_tool_cache_key("tool", {}))

        assert stats1.size == 0
        assert stats1.hits == 0

    def test_error_put_does_not_affect_stats(self) -> None:
        """Putting an error result does not change any stats."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        key = _make_tool_cache_key("tool", {})
        cache.put(key, ToolResult(output="err", error=True))

        stats = cache.stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0


# ── Clear tests ──────────────────────────────────────────────


class TestToolResultCacheClear:
    """Tests for the clear() method."""

    @staticmethod
    def _ok(output: str = "ok") -> ToolResult:
        return ToolResult(output=output, error=False)

    def test_clear_removes_entries(self) -> None:
        """Clear removes all entries and returns count."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        cache.put(_make_tool_cache_key("t1", {}), self._ok())
        cache.put(_make_tool_cache_key("t2", {}), self._ok())
        cache.put(_make_tool_cache_key("t3", {}), self._ok())

        count = cache.clear()
        assert count == 3
        assert cache.size == 0
        assert cache.get(_make_tool_cache_key("t1", {})) is None

    def test_clear_resets_stats(self) -> None:
        """Clear resets hits, misses, and evictions counters."""
        cache = ToolResultCache(default_ttl=60, max_size=2)
        k1 = _make_tool_cache_key("t1", {})
        cache.put(k1, self._ok())
        cache.get(k1)                                # hit
        cache.get(_make_tool_cache_key("miss", {}))   # miss
        cache.put(_make_tool_cache_key("t2", {}), self._ok())
        cache.put(_make_tool_cache_key("t3", {}), self._ok())  # evict

        cache.clear()
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size == 0

    def test_clear_empty_cache(self) -> None:
        """Clear on empty cache returns 0."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        assert cache.clear() == 0

    def test_repr(self) -> None:
        """repr shows useful info."""
        cache = ToolResultCache(default_ttl=60, max_size=10)
        cache.put(_make_tool_cache_key("tool", {}), self._ok())
        r = repr(cache)
        assert "1/10" in r
        assert "default_ttl=60s" in r


# ── Thread safety tests ──────────────────────────────────────


class TestToolResultCacheThreadSafety:
    """Thread safety tests for ToolResultCache."""

    @staticmethod
    def _ok(i: int = 0) -> ToolResult:
        return ToolResult(output=f"r{i}", error=False)

    def test_concurrent_puts(self) -> None:
        """Concurrent puts don't corrupt the cache."""
        cache = ToolResultCache(default_ttl=60, max_size=1000)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    key = _make_tool_cache_key(f"tool_{i}", {"i": i})
                    cache.put(key, self._ok(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 1000
        assert cache.size >= 400  # at least some survived

    def test_concurrent_reads_and_writes(self) -> None:
        """Concurrent reads and writes don't raise exceptions."""
        cache = ToolResultCache(default_ttl=60, max_size=100)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    key = _make_tool_cache_key(f"tool_{i}", {})
                    cache.put(key, self._ok(i))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(200):
                    cache.get(_make_tool_cache_key(f"tool_{i}", {}))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_clear_and_put(self) -> None:
        """Concurrent clear and put operations don't corrupt state."""
        cache = ToolResultCache(default_ttl=60, max_size=100)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    key = _make_tool_cache_key(f"tool_{i}", {})
                    cache.put(key, self._ok(i))
            except Exception as e:
                errors.append(e)

        def clearer() -> None:
            try:
                for _ in range(10):
                    cache.clear()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_get_or_none(self) -> None:
        """Concurrent get_or_none calls don't raise exceptions."""
        cache = ToolResultCache(default_ttl=60, max_size=100)
        errors: list[Exception] = []

        # Pre-populate
        for i in range(50):
            key = _make_tool_cache_key(f"tool_{i}", {"x": i})
            cache.put(key, self._ok(i))

        def reader() -> None:
            try:
                for i in range(100):
                    cache.get_or_none(f"tool_{i}", {"x": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

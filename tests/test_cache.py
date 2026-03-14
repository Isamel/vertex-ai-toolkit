"""Tests for the response cache module (vaig.core.cache).

Covers:
- Basic put/get operations
- LRU eviction
- TTL expiration
- Thread safety
- Cache key generation
- Cache stats
- Clear operation
- Edge cases (max_size=1, ttl=0)
"""

from __future__ import annotations

import hashlib
import threading
import time
from unittest.mock import patch

import pytest

from vaig.core.cache import CacheStats, ResponseCache, _make_cache_key
from vaig.core.client import GenerationResult


# ── Cache key tests ──────────────────────────────────────────


class TestMakeCacheKey:
    """Tests for _make_cache_key()."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro", "be helpful")
        key2 = _make_cache_key("hello", "gemini-2.5-pro", "be helpful")
        assert key1 == key2

    def test_different_prompts_different_keys(self) -> None:
        """Different prompts produce different keys."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro")
        key2 = _make_cache_key("world", "gemini-2.5-pro")
        assert key1 != key2

    def test_different_models_different_keys(self) -> None:
        """Different models produce different keys."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro")
        key2 = _make_cache_key("hello", "gemini-2.5-flash")
        assert key1 != key2

    def test_different_system_instructions_different_keys(self) -> None:
        """Different system instructions produce different keys."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro", "be helpful")
        key2 = _make_cache_key("hello", "gemini-2.5-pro", "be concise")
        assert key1 != key2

    def test_none_system_instruction(self) -> None:
        """None system instruction is handled consistently."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro", None)
        key2 = _make_cache_key("hello", "gemini-2.5-pro", None)
        assert key1 == key2

    def test_none_vs_empty_system_instruction(self) -> None:
        """None and empty string produce the same key (both mean 'no instruction')."""
        key1 = _make_cache_key("hello", "gemini-2.5-pro", None)
        key2 = _make_cache_key("hello", "gemini-2.5-pro", "")
        assert key1 == key2

    def test_key_is_sha256_hex(self) -> None:
        """Key is a valid SHA-256 hex digest (64 chars)."""
        key = _make_cache_key("test", "model")
        assert len(key) == 64
        # Verify it's valid hex
        int(key, 16)

    def test_key_matches_manual_sha256(self) -> None:
        """Key matches a manually computed SHA-256."""
        raw = "model:gemini-2.5-pro\nsystem:\nprompt:hello"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert _make_cache_key("hello", "gemini-2.5-pro") == expected


# ── ResponseCache tests ──────────────────────────────────────


class TestResponseCache:
    """Tests for the ResponseCache class."""

    @staticmethod
    def _make_result(text: str = "response", model: str = "gemini-2.5-pro") -> GenerationResult:
        """Helper to create a GenerationResult for testing."""
        return GenerationResult(
            text=text,
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="STOP",
        )

    def test_put_and_get(self) -> None:
        """Basic put/get cycle returns the cached value."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        result = self._make_result()
        cache.put("key1", result)

        cached = cache.get("key1")
        assert cached is result

    def test_get_miss(self) -> None:
        """Get returns None on cache miss."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_size(self) -> None:
        """Size reflects the number of entries."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        assert cache.size == 0

        cache.put("k1", self._make_result("r1"))
        assert cache.size == 1

        cache.put("k2", self._make_result("r2"))
        assert cache.size == 2

    def test_lru_eviction(self) -> None:
        """LRU eviction removes the least-recently-used entry when at capacity."""
        cache = ResponseCache(max_size=2, ttl_seconds=60)

        cache.put("k1", self._make_result("r1"))
        cache.put("k2", self._make_result("r2"))
        assert cache.size == 2

        # Adding k3 should evict k1 (oldest)
        cache.put("k3", self._make_result("r3"))
        assert cache.size == 2
        assert cache.get("k1") is None  # evicted
        assert cache.get("k2") is not None
        assert cache.get("k3") is not None

    def test_lru_access_refreshes_order(self) -> None:
        """Accessing an entry moves it to the end, changing eviction order."""
        cache = ResponseCache(max_size=2, ttl_seconds=60)

        cache.put("k1", self._make_result("r1"))
        cache.put("k2", self._make_result("r2"))

        # Access k1 — makes it most-recently used
        cache.get("k1")

        # Adding k3 should now evict k2 (older than k1 after the access)
        cache.put("k3", self._make_result("r3"))
        assert cache.get("k1") is not None  # survived
        assert cache.get("k2") is None  # evicted
        assert cache.get("k3") is not None

    def test_ttl_expiration(self) -> None:
        """Entries expire after TTL seconds."""
        cache = ResponseCache(max_size=10, ttl_seconds=1)
        cache.put("key", self._make_result())

        # Should be available immediately
        assert cache.get("key") is not None

        # Simulate TTL expiration by patching time.monotonic
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 2):
            assert cache.get("key") is None

    def test_ttl_zero_means_no_expiration(self) -> None:
        """TTL of 0 means entries never expire."""
        cache = ResponseCache(max_size=10, ttl_seconds=0)
        cache.put("key", self._make_result())

        # Even with a large time jump, entry should persist
        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 999999):
            assert cache.get("key") is not None

    def test_clear(self) -> None:
        """Clear removes all entries and returns count."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("k1", self._make_result("r1"))
        cache.put("k2", self._make_result("r2"))
        cache.put("k3", self._make_result("r3"))

        count = cache.clear()
        assert count == 3
        assert cache.size == 0
        assert cache.get("k1") is None

    def test_clear_empty_cache(self) -> None:
        """Clear on empty cache returns 0."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        assert cache.clear() == 0

    def test_update_existing_key(self) -> None:
        """Putting an existing key updates the value."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)

        r1 = self._make_result("first")
        r2 = self._make_result("second")

        cache.put("key", r1)
        cache.put("key", r2)

        assert cache.size == 1  # no duplicate
        cached = cache.get("key")
        assert cached is r2  # updated value

    def test_max_size_1(self) -> None:
        """Cache with max_size=1 always holds at most 1 entry."""
        cache = ResponseCache(max_size=1, ttl_seconds=60)

        cache.put("k1", self._make_result("r1"))
        assert cache.size == 1

        cache.put("k2", self._make_result("r2"))
        assert cache.size == 1
        assert cache.get("k1") is None
        assert cache.get("k2") is not None

    def test_invalid_max_size(self) -> None:
        """max_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ResponseCache(max_size=0, ttl_seconds=60)

    def test_invalid_ttl(self) -> None:
        """Negative ttl_seconds raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be >= 0"):
            ResponseCache(max_size=10, ttl_seconds=-1)


class TestCacheStats:
    """Tests for cache statistics tracking."""

    @staticmethod
    def _make_result() -> GenerationResult:
        return GenerationResult(text="r", model="m", usage={}, finish_reason="STOP")

    def test_initial_stats(self) -> None:
        """Fresh cache has zero stats."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size == 0
        assert stats.max_size == 10
        assert stats.hit_rate == 0.0

    def test_miss_increments(self) -> None:
        """Cache miss increments miss counter."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.get("nonexistent")
        cache.get("also_nonexistent")

        stats = cache.stats()
        assert stats.misses == 2
        assert stats.hits == 0

    def test_hit_increments(self) -> None:
        """Cache hit increments hit counter."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key", self._make_result())

        cache.get("key")
        cache.get("key")

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 0

    def test_hit_rate(self) -> None:
        """Hit rate is computed correctly."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key", self._make_result())

        cache.get("key")  # hit
        cache.get("miss")  # miss

        stats = cache.stats()
        assert stats.hit_rate == pytest.approx(0.5)

    def test_eviction_count(self) -> None:
        """Eviction counter tracks LRU evictions."""
        cache = ResponseCache(max_size=2, ttl_seconds=60)

        cache.put("k1", self._make_result())
        cache.put("k2", self._make_result())
        cache.put("k3", self._make_result())  # evicts k1

        stats = cache.stats()
        assert stats.evictions == 1

    def test_ttl_expiry_counts_as_eviction_and_miss(self) -> None:
        """TTL-expired entry counts as a miss and eviction."""
        cache = ResponseCache(max_size=10, ttl_seconds=1)
        cache.put("key", self._make_result())

        with patch("vaig.core.cache.time.monotonic", return_value=time.monotonic() + 2):
            cache.get("key")  # expired

        stats = cache.stats()
        assert stats.misses == 1
        assert stats.evictions == 1

    def test_stats_is_frozen_snapshot(self) -> None:
        """Stats returns a frozen dataclass — not a live view."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        stats1 = cache.stats()

        cache.put("key", self._make_result())
        cache.get("key")

        # stats1 should NOT have changed
        assert stats1.size == 0
        assert stats1.hits == 0

    def test_repr(self) -> None:
        """repr shows useful info."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key", self._make_result())
        r = repr(cache)
        assert "1/10" in r
        assert "ttl=60s" in r


class TestCacheThreadSafety:
    """Thread safety tests for ResponseCache."""

    @staticmethod
    def _make_result(i: int = 0) -> GenerationResult:
        return GenerationResult(text=f"r{i}", model="m", usage={}, finish_reason="STOP")

    def test_concurrent_puts(self) -> None:
        """Concurrent puts don't corrupt the cache."""
        cache = ResponseCache(max_size=1000, ttl_seconds=60)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    cache.put(f"key_{i}", self._make_result(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 1000
        assert cache.size >= 400  # At least some survived

    def test_concurrent_reads_and_writes(self) -> None:
        """Concurrent reads and writes don't raise exceptions."""
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    cache.put(f"key_{i}", self._make_result(i))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(200):
                    cache.get(f"key_{i}")
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
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    cache.put(f"key_{i}", self._make_result(i))
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


# ── CacheConfig tests ────────────────────────────────────────


class TestCacheConfig:
    """Tests for CacheConfig in Settings."""

    def test_default_disabled(self) -> None:
        """Cache is disabled by default."""
        from vaig.core.config import CacheConfig

        cfg = CacheConfig()
        assert cfg.enabled is False
        assert cfg.max_size == 128
        assert cfg.ttl_seconds == 300

    def test_cache_in_settings(self) -> None:
        """Settings includes cache config with correct defaults."""
        from vaig.core.config import Settings

        settings = Settings()
        assert settings.cache.enabled is False
        assert settings.cache.max_size == 128
        assert settings.cache.ttl_seconds == 300

    def test_cache_custom_values(self) -> None:
        """CacheConfig accepts custom values."""
        from vaig.core.config import CacheConfig

        cfg = CacheConfig(enabled=True, max_size=64, ttl_seconds=600)
        assert cfg.enabled is True
        assert cfg.max_size == 64
        assert cfg.ttl_seconds == 600


# ── GeminiClient cache integration tests ─────────────────────


class TestGeminiClientCacheIntegration:
    """Tests for cache integration in GeminiClient."""

    def test_cache_disabled_by_default(self) -> None:
        """Client has no cache when config has cache.enabled=False."""
        from vaig.core.config import Settings

        settings = Settings()
        from vaig.core.client import GeminiClient

        client = GeminiClient(settings)
        assert client.cache_enabled is False
        assert client.cache_stats() is None
        assert client.clear_cache() == 0

    def test_cache_enabled(self) -> None:
        """Client initializes cache when config has cache.enabled=True."""
        from vaig.core.config import CacheConfig, Settings

        settings = Settings(cache=CacheConfig(enabled=True, max_size=64, ttl_seconds=120))
        from vaig.core.client import GeminiClient

        client = GeminiClient(settings)
        assert client.cache_enabled is True

        stats = client.cache_stats()
        assert stats is not None
        assert stats.size == 0
        assert stats.max_size == 64

    def test_generate_caches_string_prompt_no_history(self) -> None:
        """generate() caches results for string prompts without history."""
        from unittest.mock import MagicMock, patch

        from vaig.core.config import CacheConfig, Settings

        settings = Settings(cache=CacheConfig(enabled=True, max_size=10, ttl_seconds=60))
        from vaig.core.client import GeminiClient, GenerationResult

        client = GeminiClient(settings)
        client._initialized = True

        # Mock the retry_with_backoff to return a result directly
        result = GenerationResult(
            text="cached response",
            model="gemini-2.5-pro",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="STOP",
        )

        with patch.object(client, "_retry_with_backoff", return_value=result):
            # First call — should invoke _retry_with_backoff
            r1 = client.generate("what is AI?", system_instruction="be helpful")
            assert r1.text == "cached response"

        # Second call — should return cached result (no _retry_with_backoff needed)
        r2 = client.generate("what is AI?", system_instruction="be helpful")
        assert r2.text == "cached response"
        assert r2 is r1  # Same object (from cache)

        stats = client.cache_stats()
        assert stats is not None
        assert stats.hits == 1
        assert stats.misses == 1  # first call was a miss

    def test_generate_does_not_cache_with_history(self) -> None:
        """generate() does NOT cache when history is provided."""
        from unittest.mock import MagicMock, patch

        from vaig.core.config import CacheConfig, Settings

        settings = Settings(cache=CacheConfig(enabled=True, max_size=10, ttl_seconds=60))
        from vaig.core.client import ChatMessage, GeminiClient, GenerationResult

        client = GeminiClient(settings)
        client._initialized = True

        result = GenerationResult(
            text="response",
            model="gemini-2.5-pro",
            usage={},
            finish_reason="STOP",
        )

        history = [ChatMessage(role="user", content="previous question")]
        call_count = 0

        def side_effect(fn, **kwargs):
            nonlocal call_count
            call_count += 1
            return result

        with patch.object(client, "_retry_with_backoff", side_effect=side_effect):
            client.generate("follow-up", history=history)
            client.generate("follow-up", history=history)

        # Both calls should have gone through (no caching with history)
        assert call_count == 2

        stats = client.cache_stats()
        assert stats is not None
        assert stats.hits == 0  # no cache activity for history-based queries

    def test_generate_does_not_cache_multimodal_prompt(self) -> None:
        """generate() does NOT cache when prompt is a list of Parts."""
        from unittest.mock import MagicMock, patch

        from google.genai import types

        from vaig.core.config import CacheConfig, Settings

        settings = Settings(cache=CacheConfig(enabled=True, max_size=10, ttl_seconds=60))
        from vaig.core.client import GeminiClient, GenerationResult

        client = GeminiClient(settings)
        client._initialized = True

        result = GenerationResult(text="response", model="m", usage={}, finish_reason="STOP")
        call_count = 0

        def side_effect(fn, **kwargs):
            nonlocal call_count
            call_count += 1
            return result

        multimodal_prompt = [types.Part.from_text(text="describe this image")]
        with patch.object(client, "_retry_with_backoff", side_effect=side_effect):
            client.generate(multimodal_prompt)
            client.generate(multimodal_prompt)

        assert call_count == 2  # no caching for multimodal

    def test_clear_cache_via_client(self) -> None:
        """clear_cache() removes cached entries."""
        from unittest.mock import patch

        from vaig.core.config import CacheConfig, Settings

        settings = Settings(cache=CacheConfig(enabled=True, max_size=10, ttl_seconds=60))
        from vaig.core.client import GeminiClient, GenerationResult

        client = GeminiClient(settings)
        client._initialized = True

        result = GenerationResult(text="resp", model="m", usage={}, finish_reason="STOP")
        with patch.object(client, "_retry_with_backoff", return_value=result):
            client.generate("q1")
            client.generate("q2")

        assert client.cache_stats().size == 2  # type: ignore[union-attr]
        cleared = client.clear_cache()
        assert cleared == 2
        assert client.cache_stats().size == 0  # type: ignore[union-attr]

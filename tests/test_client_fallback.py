"""Tests for SPEC-RATE-04: GeminiClient mid-call model fallback.

Scenarios tested:
  S1 — persistent 429s on primary → switch to fallback model, complete
  S2 — fallback_model=None → 429s exhaust retries, GeminiRateLimitError raised
  S3 — _fallback_active=True already → no double-fallback
  S4 — non-429 error → no fallback triggered
  S5 — async path mirrors sync (S1 for _async_retry_with_backoff)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from vaig.core.client import GeminiClient
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelsConfig,
    RetryConfig,
    Settings,
)
from vaig.core.exceptions import GeminiRateLimitError

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Fast-retry settings with a fallback model configured (max_retries=6)."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(),
        models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        retry=RetryConfig(
            max_retries=6,
            initial_delay=0.01,
            max_delay=0.05,
            backoff_multiplier=2.0,
            retryable_status_codes=[429, 500, 502, 503, 504],
            rate_limit_initial_delay=0.01,
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    """GeminiClient constructed with fallback_model="gemini-2.5-flash"."""
    return GeminiClient(settings, fallback_model="gemini-2.5-flash")


@pytest.fixture()
def client_no_fallback(settings: Settings) -> GeminiClient:
    """GeminiClient constructed with fallback_model=None."""
    return GeminiClient(settings, fallback_model=None)


# ── S1: Fallback activates on persistent 429s (sync) ─────────


class TestSyncFallbackActivation:
    """S1 — 429s at attempt >= max_retries // 2 trigger model switch."""

    @patch("vaig.core.client.time.sleep")
    def test_fallback_activates_after_half_retries(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """429 on attempts 0-3 (attempt 3 >= max_retries//2=3) → fallback fires, succeeds on call 5."""
        # max_retries=6, so max_retries//2=3. Fallback triggers when attempt >= 3.
        # Attempts are 0-indexed; call N is attempt N-1.
        # 429 raised on calls 1-4 (attempts 0-3); success on call 5 (attempt 4).
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "success"

        result = client._retry_with_backoff(fn)

        assert result == "success"
        assert client.fallback_active is True

    @patch("vaig.core.client.time.sleep")
    def test_fallback_active_property_true_after_switch(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """After fallback fires, client.fallback_active is True."""
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "ok"

        client._retry_with_backoff(fn)
        assert client.fallback_active is True


# ── S2: No fallback when fallback_model=None ─────────────────


class TestSyncNoFallbackWhenNone:
    """S2 — fallback_model=None → retries exhaust normally."""

    @patch("vaig.core.client.time.sleep")
    def test_no_fallback_when_none_raises_rate_limit_error(
        self,
        mock_sleep: MagicMock,
        client_no_fallback: GeminiClient,
        settings: Settings,
    ) -> None:
        """With fallback_model=None, GeminiRateLimitError raised after max_retries."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(429, "Resource exhausted"),
        )

        with pytest.raises(GeminiRateLimitError):
            client_no_fallback._retry_with_backoff(fn)

        # Called max_retries + 1 times (initial + retries)
        assert fn.call_count == settings.retry.max_retries + 1

    @patch("vaig.core.client.time.sleep")
    def test_no_model_switch_when_fallback_none(
        self,
        mock_sleep: MagicMock,
        client_no_fallback: GeminiClient,
    ) -> None:
        """fallback_active stays False when fallback_model is None."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(429, "Resource exhausted"),
        )

        with pytest.raises(GeminiRateLimitError):
            client_no_fallback._retry_with_backoff(fn)

        assert client_no_fallback.fallback_active is False


# ── S3: No double-fallback ────────────────────────────────────


class TestSyncNoDoubleFallback:
    """S3 — if _fallback_active=True, no second model switch."""

    @patch("vaig.core.client.time.sleep")
    def test_no_double_fallback_when_already_active(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """_switch_model is a no-op when _fallback_active is already True."""
        # Pre-set fallback as already active
        client._fallback_active = True
        client._current_model_id = "gemini-2.5-flash"
        original_model = client._current_model_id

        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "ok"

        result = client._retry_with_backoff(fn)

        assert result == "ok"
        # Model should NOT have changed again
        assert client._current_model_id == original_model


# ── S4: Non-429 error does not trigger fallback ───────────────


class TestSyncNon429NoFallback:
    """S4 — 500 error does not activate fallback."""

    def test_500_does_not_trigger_fallback(
        self,
        client: GeminiClient,
    ) -> None:
        """Non-429 error (500) → no fallback, GeminiConnectionError raised."""
        from vaig.core.exceptions import GeminiConnectionError

        fn = MagicMock(
            side_effect=genai_errors.ServerError(500, "Internal server error"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        assert client.fallback_active is False

    def test_502_does_not_trigger_fallback(
        self,
        client: GeminiClient,
    ) -> None:
        """Non-429 error (502) → no fallback."""
        from vaig.core.exceptions import GeminiConnectionError

        fn = MagicMock(
            side_effect=genai_errors.ServerError(502, "Bad gateway"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        assert client.fallback_active is False


# ── S5: Async path mirrors sync (S1 for _async_retry_with_backoff) ─


class TestAsyncFallbackActivation:
    """S5 — async path mirrors sync: fallback activates on persistent 429s."""

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_fallback_activates_after_half_retries(
        self,
        mock_sleep: AsyncMock,
        client: GeminiClient,
    ) -> None:
        """Async: 429 on attempts 0-3 → switches to flash at attempt >= 3 → succeeds."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "success"

        result = await client._async_retry_with_backoff(fn)

        assert result == "success"
        assert client.fallback_active is True

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_no_fallback_when_none_raises_rate_limit_error(
        self,
        mock_sleep: AsyncMock,
        client_no_fallback: GeminiClient,
        settings: Settings,
    ) -> None:
        """Async: fallback_model=None → GeminiRateLimitError after max_retries."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ClientError(429, "Resource exhausted")

        with pytest.raises(GeminiRateLimitError):
            await client_no_fallback._async_retry_with_backoff(fn)

        assert call_count == settings.retry.max_retries + 1

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_no_fallback_when_none_stays_false(
        self,
        mock_sleep: AsyncMock,
        client_no_fallback: GeminiClient,
    ) -> None:
        """Async: fallback_active remains False when no fallback configured."""

        async def fn() -> str:
            raise genai_errors.ClientError(429, "Resource exhausted")

        with pytest.raises(GeminiRateLimitError):
            await client_no_fallback._async_retry_with_backoff(fn)

        assert client_no_fallback.fallback_active is False

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_500_does_not_trigger_fallback(
        self,
        mock_sleep: AsyncMock,
        client: GeminiClient,
    ) -> None:
        """Async: non-429 error (500) → no fallback activated."""
        from vaig.core.exceptions import GeminiConnectionError

        async def fn() -> str:
            raise genai_errors.ServerError(500, "Internal server error")

        with pytest.raises(GeminiConnectionError):
            await client._async_retry_with_backoff(fn)

        assert client.fallback_active is False

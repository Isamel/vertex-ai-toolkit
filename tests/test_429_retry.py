"""Tests for two-layer retry: SDK-level HttpRetryOptions + app-level genai_errors.APIError handling.

Layer 1 — SDK retries transient HTTP errors automatically via ``HttpRetryOptions``.
Layer 2 — vaig catches ``genai_errors.APIError``:
  - 429 (rate-limit): retried at app level with ``rate_limit_initial_delay`` backoff.
  - Other retryable codes (500, 502, 503, 504): NOT retried at app level
    (the SDK already exhausted its retry budget).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from vaig.core.client import _RETRYABLE_STATUS_CODES, GeminiClient
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelsConfig,
    RetryConfig,
    Settings,
)
from vaig.core.exceptions import (
    GeminiConnectionError,
    GeminiRateLimitError,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Settings with a fast retry config (no real delays in tests)."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(),
        models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        retry=RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.05,
            backoff_multiplier=2.0,
            retryable_status_codes=[429, 500, 502, 503, 504],
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    """An uninitialized GeminiClient with fast retry config."""
    return GeminiClient(settings)


# ── Module-level constant ────────────────────────────────────


class TestRetryableStatusCodes:
    """Tests for the _RETRYABLE_STATUS_CODES module constant."""

    def test_retryable_status_codes_is_frozenset(self) -> None:
        """_RETRYABLE_STATUS_CODES is a frozenset of expected codes."""
        assert isinstance(_RETRYABLE_STATUS_CODES, frozenset)
        assert frozenset({429, 500, 502, 503, 504}) == _RETRYABLE_STATUS_CODES


# ── Layer 1: SDK-level HttpRetryOptions ──────────────────────


class TestBuildHttpOptions:
    """Tests for _build_http_options mapping RetryConfig → HttpRetryOptions."""

    def test_build_http_options_uses_retry_config(self, client: GeminiClient, settings: Settings) -> None:
        """_build_http_options maps RetryConfig fields to HttpRetryOptions."""
        http_opts = client._build_http_options()

        retry_cfg = settings.retry
        assert http_opts.retry_options is not None
        retry_opts = http_opts.retry_options
        # SDK counts initial call as attempt, so attempts = max_retries + 1
        assert retry_opts.attempts == retry_cfg.max_retries + 1
        assert retry_opts.initial_delay == retry_cfg.initial_delay
        assert retry_opts.max_delay == retry_cfg.max_delay
        assert retry_opts.exp_base == retry_cfg.backoff_multiplier
        assert retry_opts.jitter == 0.5
        assert retry_opts.http_status_codes == retry_cfg.retryable_status_codes

    @patch("vaig.core.client.get_credentials")
    @patch("vaig.core.client.genai.Client")
    def test_genai_client_receives_http_options(
        self,
        mock_genai_client: MagicMock,
        mock_get_credentials: MagicMock,
        client: GeminiClient,
    ) -> None:
        """genai.Client() is called with http_options from _build_http_options."""
        mock_get_credentials.return_value = MagicMock()

        client.initialize()

        mock_genai_client.assert_called_once()
        call_kwargs = mock_genai_client.call_args[1]
        assert "http_options" in call_kwargs
        http_opts = call_kwargs["http_options"]
        assert http_opts.retry_options is not None
        assert http_opts.retry_options.http_status_codes == [429, 500, 502, 503, 504]


# ── Layer 2: App-level genai_errors.APIError handling (sync) ─
#
# The SDK already retried via HttpRetryOptions.  For 429, vaig's handler
# DOES retry at app level with a longer backoff.  For other retryable codes
# (500, 502, etc.) it does NOT retry — catches, logs, and converts.


class TestSyncGenaiErrorHandling:
    """Tests for _retry_with_backoff catching genai_errors.APIError."""

    def test_genai_429_retries_at_app_level(
        self,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """A genai ClientError 429 IS retried at app level with rate_limit backoff."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(429, "Resource exhausted"),
        )

        with pytest.raises(GeminiRateLimitError):
            client._retry_with_backoff(fn)

        # fn called max_retries + 1 times (initial + retries)
        assert fn.call_count == settings.retry.max_retries + 1

    def test_genai_500_converts_to_connection_error_without_retry(
        self,
        client: GeminiClient,
    ) -> None:
        """A genai ServerError 500 is caught and converted to GeminiConnectionError — NOT retried."""
        fn = MagicMock(
            side_effect=genai_errors.ServerError(500, "Internal server error"),
        )

        with pytest.raises(GeminiConnectionError) as exc_info:
            client._retry_with_backoff(fn)

        assert fn.call_count == 1
        assert isinstance(exc_info.value.original_error, genai_errors.ServerError)

    def test_genai_502_converts_to_connection_error(
        self,
        client: GeminiClient,
    ) -> None:
        """A genai ServerError 502 is caught and converted to GeminiConnectionError."""
        fn = MagicMock(
            side_effect=genai_errors.ServerError(502, "Bad gateway"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        assert fn.call_count == 1

    def test_genai_400_propagates_immediately(
        self,
        client: GeminiClient,
    ) -> None:
        """A genai ClientError with non-retryable code 400 propagates immediately."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(400, "Bad request"),
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            client._retry_with_backoff(fn)

        assert exc_info.value.code == 400
        assert fn.call_count == 1  # No retries

    @patch("vaig.core.client.time.sleep")
    def test_genai_400_cancelled_retries_with_backoff(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """A genai ClientError 400 CANCELLED IS retried at app level (transient)."""
        fn = MagicMock(
            side_effect=[
                genai_errors.ClientError(400, "400 CANCELLED. The operation was cancelled."),
                "success",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "success"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("vaig.core.client.time.sleep")
    def test_genai_400_cancelled_exhausts_retries(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """A genai ClientError 400 CANCELLED that persists raises GeminiConnectionError after all retries."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(400, "400 CANCELLED. The operation was cancelled."),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        assert fn.call_count == settings.retry.max_retries + 1

    def test_genai_403_propagates_immediately(
        self,
        client: GeminiClient,
    ) -> None:
        """A genai ClientError with non-retryable code 403 propagates immediately."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(403, "Forbidden"),
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            client._retry_with_backoff(fn)

        assert exc_info.value.code == 403
        assert fn.call_count == 1

    @patch.object(GeminiClient, "_reinitialize_with_fallback")
    def test_genai_retryable_with_ssl_triggers_fallback(
        self,
        mock_fallback: MagicMock,
        client: GeminiClient,
    ) -> None:
        """A retryable genai error wrapping SSL triggers location fallback."""
        ssl_cause = ConnectionResetError("Connection reset by peer")
        exc = genai_errors.ServerError(503, "Service unavailable")
        exc.__cause__ = ssl_cause

        fn = MagicMock(side_effect=[exc, "success-after-fallback"])

        result = client._retry_with_backoff(fn)

        assert result == "success-after-fallback"
        mock_fallback.assert_called_once()
        # fn called twice: first attempt → error → fallback → retry once
        assert fn.call_count == 2

    def test_time_sleep_called_for_genai_429(
        self,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """Verify time.sleep IS called when genai 429 is caught (app-level retry)."""
        fn = MagicMock(
            side_effect=genai_errors.ClientError(429, "Resource exhausted"),
        )

        with patch("vaig.core.client.time.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                client._retry_with_backoff(fn)

            assert mock_sleep.call_count == settings.retry.max_retries

    def test_no_time_sleep_called_for_genai_500(
        self,
        client: GeminiClient,
    ) -> None:
        """Verify time.sleep is NOT called when genai 500 is caught (no app retry)."""
        fn = MagicMock(
            side_effect=genai_errors.ServerError(500, "Internal server error"),
        )

        with patch("vaig.core.client.time.sleep") as mock_sleep:
            with pytest.raises(GeminiConnectionError):
                client._retry_with_backoff(fn)

            mock_sleep.assert_not_called()


# ── Layer 2: App-level genai_errors.APIError handling (async) ─


class TestAsyncGenaiErrorHandling:
    """Tests for _async_retry_with_backoff catching genai_errors.APIError."""

    @pytest.mark.asyncio()
    async def test_async_genai_429_retries_at_app_level(
        self,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """Async: genai ClientError 429 IS retried at app level."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ClientError(429, "Resource exhausted")

        with pytest.raises(GeminiRateLimitError):
            await client._async_retry_with_backoff(fn)

        assert call_count == settings.retry.max_retries + 1

    @pytest.mark.asyncio()
    async def test_async_genai_500_converts_to_connection_error_without_retry(
        self,
        client: GeminiClient,
    ) -> None:
        """Async: genai ServerError 500 → GeminiConnectionError, no retry."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ServerError(500, "Internal server error")

        with pytest.raises(GeminiConnectionError) as exc_info:
            await client._async_retry_with_backoff(fn)

        assert call_count == 1
        assert isinstance(exc_info.value.original_error, genai_errors.ServerError)

    @pytest.mark.asyncio()
    async def test_async_genai_400_propagates_immediately(
        self,
        client: GeminiClient,
    ) -> None:
        """Async: genai ClientError 400 propagates without conversion."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ClientError(400, "Bad request")

        with pytest.raises(genai_errors.ClientError) as exc_info:
            await client._async_retry_with_backoff(fn)

        assert exc_info.value.code == 400
        assert call_count == 1

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_genai_400_cancelled_retries_with_backoff(
        self,
        mock_sleep: AsyncMock,
        client: GeminiClient,
    ) -> None:
        """Async: genai ClientError 400 CANCELLED IS retried at app level (transient)."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise genai_errors.ClientError(400, "400 CANCELLED. The operation was cancelled.")
            return "success"

        result = await client._async_retry_with_backoff(fn)

        assert result == "success"
        assert call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio()
    @patch("vaig.core.client.asyncio.sleep")
    async def test_async_genai_400_cancelled_exhausts_retries(
        self,
        mock_sleep: AsyncMock,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """Async: genai ClientError 400 CANCELLED that persists raises GeminiConnectionError after all retries."""
        async def fn() -> str:
            raise genai_errors.ClientError(400, "400 CANCELLED. The operation was cancelled.")

        with pytest.raises(GeminiConnectionError):
            await client._async_retry_with_backoff(fn)

        assert mock_sleep.call_count == settings.retry.max_retries

    @pytest.mark.asyncio()
    @patch.object(GeminiClient, "_async_reinitialize_with_fallback")
    async def test_async_genai_retryable_with_ssl_triggers_fallback(
        self,
        mock_fallback: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Async: retryable genai error wrapping SSL triggers location fallback."""
        ssl_cause = ConnectionResetError("Connection reset by peer")
        exc = genai_errors.ServerError(503, "Service unavailable")
        exc.__cause__ = ssl_cause

        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise exc
            return "success-after-fallback"

        result = await client._async_retry_with_backoff(fn)

        assert result == "success-after-fallback"
        mock_fallback.assert_called_once()
        assert call_count == 2

    @pytest.mark.asyncio()
    async def test_async_asyncio_sleep_called_for_genai_429(
        self,
        client: GeminiClient,
        settings: Settings,
    ) -> None:
        """Verify asyncio.sleep IS called when genai 429 is caught (app-level retry)."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ClientError(429, "Resource exhausted")

        with patch("vaig.core.client.asyncio.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                await client._async_retry_with_backoff(fn)

            assert mock_sleep.call_count == settings.retry.max_retries

    @pytest.mark.asyncio()
    async def test_async_no_asyncio_sleep_called_for_genai_500(
        self,
        client: GeminiClient,
    ) -> None:
        """Verify asyncio.sleep is NOT called when genai 500 is caught (no app retry)."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ServerError(500, "Internal server error")

        with patch("vaig.core.client.asyncio.sleep") as mock_sleep:
            with pytest.raises(GeminiConnectionError):
                await client._async_retry_with_backoff(fn)

            mock_sleep.assert_not_called()


# ── Rate-limit initial delay (rate_limit_initial_delay) ──────


class TestRateLimitInitialDelay:
    """Verify 429 errors use the longer ``rate_limit_initial_delay`` backoff."""

    @pytest.fixture()
    def rl_settings(self) -> Settings:
        """Settings where rate_limit_initial_delay is clearly distinct from initial_delay."""
        return Settings(
            gcp=GCPConfig(project_id="test-project", location="us-central1"),
            generation=GenerationConfig(),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
            retry=RetryConfig(
                max_retries=2,
                initial_delay=0.01,
                max_delay=120.0,
                backoff_multiplier=2.0,
                rate_limit_initial_delay=8.0,
            ),
        )

    @pytest.fixture()
    def rl_client(self, rl_settings: Settings) -> GeminiClient:
        return GeminiClient(rl_settings)

    def test_sync_genai_429_uses_rate_limit_delay(
        self,
        rl_client: GeminiClient,
    ) -> None:
        """Sync: genai 429 sleep values start at rate_limit_initial_delay (8s), not initial_delay (0.01s)."""
        fn = MagicMock(side_effect=genai_errors.ClientError(429, "Resource exhausted"))

        with patch("vaig.core.client.time.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                rl_client._retry_with_backoff(fn)

            # First sleep should be ~8.0 + jitter (not ~0.01)
            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep >= 8.0, f"Expected >=8.0 for 429, got {first_sleep}"

    def test_sync_resource_exhausted_uses_rate_limit_delay(
        self,
        rl_client: GeminiClient,
    ) -> None:
        """Sync: google.api_core ResourceExhausted (429) uses rate_limit_initial_delay."""
        from google.api_core import exceptions as google_exceptions

        fn = MagicMock(side_effect=google_exceptions.ResourceExhausted("quota exceeded"))

        with patch("vaig.core.client.time.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                rl_client._retry_with_backoff(fn)

            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep >= 8.0, f"Expected >=8.0 for ResourceExhausted, got {first_sleep}"

    @pytest.mark.asyncio()
    async def test_async_genai_429_uses_rate_limit_delay(
        self,
        rl_client: GeminiClient,
    ) -> None:
        """Async: genai 429 sleep values start at rate_limit_initial_delay."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise genai_errors.ClientError(429, "Resource exhausted")

        with patch("vaig.core.client.asyncio.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                await rl_client._async_retry_with_backoff(fn)

            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep >= 8.0, f"Expected >=8.0 for async 429, got {first_sleep}"

    @pytest.mark.asyncio()
    async def test_async_resource_exhausted_uses_rate_limit_delay(
        self,
        rl_client: GeminiClient,
    ) -> None:
        """Async: google.api_core ResourceExhausted uses rate_limit_initial_delay."""
        from google.api_core import exceptions as google_exceptions

        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise google_exceptions.ResourceExhausted("quota exceeded")

        with patch("vaig.core.client.asyncio.sleep") as mock_sleep:
            with pytest.raises(GeminiRateLimitError):
                await rl_client._async_retry_with_backoff(fn)

            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep >= 8.0, f"Expected >=8.0 for async ResourceExhausted, got {first_sleep}"

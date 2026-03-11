"""Tests for retry logic, custom exceptions, RetryConfig, and location fallback."""

from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from vaig.core.client import GeminiClient, _RETRYABLE_EXCEPTIONS, _is_ssl_or_connection_error
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelsConfig,
    RetryConfig,
    Settings,
    reset_settings,
)
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
    VAIGError,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset the settings singleton between tests."""
    reset_settings()


@pytest.fixture()
def settings() -> Settings:
    """Settings with a fast retry config (no real delays in tests)."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(),
        models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        retry=RetryConfig(
            max_retries=3,
            initial_delay=0.01,  # Near-instant for tests
            max_delay=0.05,
            backoff_multiplier=2.0,
            retryable_status_codes=[429, 500, 502, 503, 504],
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    """An uninitialized GeminiClient with fast retry config."""
    return GeminiClient(settings)


# ── TestExceptionHierarchy ───────────────────────────────────


class TestExceptionHierarchy:
    """Tests for the custom exception class hierarchy."""

    def test_vaig_error_is_base_exception(self) -> None:
        assert issubclass(VAIGError, Exception)

    def test_gemini_client_error_extends_vaig_error(self) -> None:
        assert issubclass(GeminiClientError, VAIGError)

    def test_rate_limit_error_extends_client_error(self) -> None:
        assert issubclass(GeminiRateLimitError, GeminiClientError)

    def test_connection_error_extends_client_error(self) -> None:
        assert issubclass(GeminiConnectionError, GeminiClientError)

    def test_gemini_client_error_stores_metadata(self) -> None:
        original = ValueError("boom")
        err = GeminiClientError(
            "All retries exhausted",
            original_error=original,
            retries_attempted=3,
        )
        assert str(err) == "All retries exhausted"
        assert err.original_error is original
        assert err.retries_attempted == 3

    def test_gemini_client_error_defaults(self) -> None:
        err = GeminiClientError("simple error")
        assert err.original_error is None
        assert err.retries_attempted == 0

    def test_rate_limit_error_stores_metadata(self) -> None:
        original = google_exceptions.ResourceExhausted("429 quota exceeded")
        err = GeminiRateLimitError(
            "Rate limited",
            original_error=original,
            retries_attempted=3,
        )
        assert isinstance(err, GeminiClientError)
        assert isinstance(err, VAIGError)
        assert err.retries_attempted == 3
        assert err.original_error is original

    def test_connection_error_stores_metadata(self) -> None:
        original = google_exceptions.ServiceUnavailable("503")
        err = GeminiConnectionError(
            "Server down",
            original_error=original,
            retries_attempted=2,
        )
        assert isinstance(err, GeminiClientError)
        assert err.retries_attempted == 2

    def test_catch_vaig_error_catches_all_subtypes(self) -> None:
        """A single `except VAIGError` should catch all custom exceptions."""
        for exc_cls in (GeminiClientError, GeminiRateLimitError, GeminiConnectionError):
            with pytest.raises(VAIGError):
                raise exc_cls("test")


# ── TestRetryConfig ──────────────────────────────────────────


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_default_values(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.initial_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_multiplier == 2.0
        assert cfg.retryable_status_codes == [429, 500, 502, 503, 504]

    def test_custom_values(self) -> None:
        cfg = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=120.0,
            backoff_multiplier=3.0,
            retryable_status_codes=[429, 503],
        )
        assert cfg.max_retries == 5
        assert cfg.initial_delay == 0.5
        assert cfg.max_delay == 120.0
        assert cfg.backoff_multiplier == 3.0
        assert cfg.retryable_status_codes == [429, 503]

    def test_zero_retries_means_single_attempt(self) -> None:
        cfg = RetryConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_retry_config_in_settings(self) -> None:
        settings = Settings(retry=RetryConfig(max_retries=7))
        assert settings.retry.max_retries == 7


# ── TestRetryableExceptions ──────────────────────────────────


class TestRetryableExceptions:
    """Tests that the right Google exceptions are considered retryable."""

    def test_resource_exhausted_is_retryable(self) -> None:
        assert google_exceptions.ResourceExhausted in _RETRYABLE_EXCEPTIONS

    def test_service_unavailable_is_retryable(self) -> None:
        assert google_exceptions.ServiceUnavailable in _RETRYABLE_EXCEPTIONS

    def test_internal_server_error_is_retryable(self) -> None:
        assert google_exceptions.InternalServerError in _RETRYABLE_EXCEPTIONS

    def test_deadline_exceeded_is_retryable(self) -> None:
        assert google_exceptions.DeadlineExceeded in _RETRYABLE_EXCEPTIONS

    def test_aborted_is_retryable(self) -> None:
        assert google_exceptions.Aborted in _RETRYABLE_EXCEPTIONS

    def test_not_found_is_not_retryable(self) -> None:
        assert google_exceptions.NotFound not in _RETRYABLE_EXCEPTIONS

    def test_permission_denied_is_not_retryable(self) -> None:
        assert google_exceptions.PermissionDenied not in _RETRYABLE_EXCEPTIONS

    def test_invalid_argument_is_not_retryable(self) -> None:
        assert google_exceptions.InvalidArgument not in _RETRYABLE_EXCEPTIONS


# ── TestRetryWithBackoff ─────────────────────────────────────


class TestRetryWithBackoff:
    """Tests for GeminiClient._retry_with_backoff()."""

    @patch("vaig.core.client.time.sleep")
    def test_succeeds_on_first_attempt(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(return_value="success")

        result = client._retry_with_backoff(fn)

        assert result == "success"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("vaig.core.client.time.sleep")
    def test_retries_on_resource_exhausted(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=[
                google_exceptions.ResourceExhausted("429"),
                google_exceptions.ResourceExhausted("429"),
                "success",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "success"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("vaig.core.client.time.sleep")
    def test_retries_on_service_unavailable(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=[
                google_exceptions.ServiceUnavailable("503"),
                "success",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "success"
        assert fn.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("vaig.core.client.time.sleep")
    def test_retries_on_internal_server_error(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=[
                google_exceptions.InternalServerError("500"),
                "ok",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "ok"
        assert fn.call_count == 2

    @patch("vaig.core.client.time.sleep")
    def test_retries_on_deadline_exceeded(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=[
                google_exceptions.DeadlineExceeded("504"),
                "ok",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "ok"

    @patch("vaig.core.client.time.sleep")
    def test_raises_rate_limit_error_after_exhausting_retries_on_429(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When all retries fail with 429, should raise GeminiRateLimitError."""
        fn = MagicMock(
            side_effect=google_exceptions.ResourceExhausted("quota exceeded"),
        )

        with pytest.raises(GeminiRateLimitError) as exc_info:
            client._retry_with_backoff(fn)

        assert fn.call_count == 4  # 1 initial + 3 retries
        assert exc_info.value.retries_attempted == 3
        assert isinstance(exc_info.value.original_error, google_exceptions.ResourceExhausted)
        assert "All 3 retries exhausted" in str(exc_info.value)

    @patch("vaig.core.client.time.sleep")
    def test_raises_connection_error_after_exhausting_retries_on_503(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When all retries fail with 503, should raise GeminiConnectionError."""
        fn = MagicMock(
            side_effect=google_exceptions.ServiceUnavailable("service down"),
        )

        with pytest.raises(GeminiConnectionError) as exc_info:
            client._retry_with_backoff(fn)

        assert fn.call_count == 4
        assert exc_info.value.retries_attempted == 3
        assert isinstance(exc_info.value.original_error, google_exceptions.ServiceUnavailable)

    @patch("vaig.core.client.time.sleep")
    def test_raises_connection_error_after_exhausting_retries_on_500(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=google_exceptions.InternalServerError("internal error"),
        )

        with pytest.raises(GeminiConnectionError) as exc_info:
            client._retry_with_backoff(fn)

        assert exc_info.value.retries_attempted == 3
        assert isinstance(exc_info.value.original_error, google_exceptions.InternalServerError)

    @patch("vaig.core.client.time.sleep")
    def test_non_retryable_error_raises_immediately(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Non-retryable exceptions should NOT be caught — they bubble up."""
        fn = MagicMock(
            side_effect=google_exceptions.NotFound("404 not found"),
        )

        with pytest.raises(google_exceptions.NotFound):
            client._retry_with_backoff(fn)

        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("vaig.core.client.time.sleep")
    def test_permission_denied_raises_immediately(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(
            side_effect=google_exceptions.PermissionDenied("forbidden"),
        )

        with pytest.raises(google_exceptions.PermissionDenied):
            client._retry_with_backoff(fn)

        fn.assert_called_once()

    @patch("vaig.core.client.time.sleep")
    def test_generic_exception_raises_immediately(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        fn = MagicMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            client._retry_with_backoff(fn)

        fn.assert_called_once()

    @patch("vaig.core.client.time.sleep")
    def test_zero_retries_means_single_attempt(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """With max_retries=0, only one attempt should be made."""
        settings.retry.max_retries = 0
        client = GeminiClient(settings)
        fn = MagicMock(
            side_effect=google_exceptions.ResourceExhausted("429"),
        )

        with pytest.raises(GeminiRateLimitError) as exc_info:
            client._retry_with_backoff(fn)

        fn.assert_called_once()
        assert exc_info.value.retries_attempted == 0
        mock_sleep.assert_not_called()

    @patch("vaig.core.client.time.sleep")
    def test_backoff_delay_increases_exponentially(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """Verify that sleep times follow exponential backoff."""
        settings.retry.initial_delay = 1.0
        settings.retry.max_delay = 100.0
        settings.retry.backoff_multiplier = 2.0
        settings.retry.max_retries = 3
        client = GeminiClient(settings)

        fn = MagicMock(
            side_effect=google_exceptions.ServiceUnavailable("503"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        # 3 retries = 3 sleeps
        assert mock_sleep.call_count == 3
        sleep_times = [call.args[0] for call in mock_sleep.call_args_list]
        # Delays: 1.0+jitter, 2.0+jitter, 4.0+jitter (jitter in [0, 0.5])
        assert 1.0 <= sleep_times[0] <= 1.5
        assert 2.0 <= sleep_times[1] <= 2.5
        assert 4.0 <= sleep_times[2] <= 4.5

    @patch("vaig.core.client.time.sleep")
    def test_delay_capped_at_max_delay(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """Sleep time should never exceed max_delay + jitter."""
        settings.retry.initial_delay = 100.0
        settings.retry.max_delay = 5.0
        settings.retry.max_retries = 1
        client = GeminiClient(settings)

        fn = MagicMock(
            side_effect=google_exceptions.ServiceUnavailable("503"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn)

        sleep_time = mock_sleep.call_args.args[0]
        # max_delay is 5.0, jitter adds up to 0.5
        assert sleep_time <= 5.5

    @patch("vaig.core.client.time.sleep")
    def test_mixed_retryable_errors_still_retry(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Different retryable errors across attempts should all be retried."""
        fn = MagicMock(
            side_effect=[
                google_exceptions.ResourceExhausted("429"),
                google_exceptions.ServiceUnavailable("503"),
                google_exceptions.InternalServerError("500"),
                "finally ok",
            ],
        )

        result = client._retry_with_backoff(fn)

        assert result == "finally ok"
        assert fn.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("vaig.core.client.time.monotonic")
    @patch("vaig.core.client.time.sleep")
    def test_timeout_stops_retries(
        self,
        mock_sleep: MagicMock,
        mock_monotonic: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Wall-clock timeout should stop the retry loop early."""
        # Simulate time: start=0, check at attempt 2: elapsed=10s (past 5s timeout)
        mock_monotonic.side_effect = [0.0, 10.0]

        fn = MagicMock(
            side_effect=google_exceptions.ServiceUnavailable("503"),
        )

        with pytest.raises(GeminiConnectionError):
            client._retry_with_backoff(fn, timeout=5.0)

        # Only 1 attempt (first call), then timeout hit before second attempt
        fn.assert_called_once()

    @patch("vaig.core.client.time.monotonic")
    @patch("vaig.core.client.time.sleep")
    def test_timeout_none_means_no_time_limit(
        self,
        mock_sleep: MagicMock,
        mock_monotonic: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When timeout is None, time.monotonic should not be used for checking."""
        fn = MagicMock(
            side_effect=[
                google_exceptions.ServiceUnavailable("503"),
                "ok",
            ],
        )

        result = client._retry_with_backoff(fn, timeout=None)

        assert result == "ok"
        # monotonic IS called once for start_time check (but only if timeout is not None)
        # With timeout=None, monotonic should not be called at all
        mock_monotonic.assert_not_called()

    @patch("vaig.core.client.time.sleep")
    def test_chained_exception_preserves_cause(
        self,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """The __cause__ of the raised exception should be the original error."""
        original = google_exceptions.ResourceExhausted("quota exceeded")
        fn = MagicMock(side_effect=original)

        with pytest.raises(GeminiRateLimitError) as exc_info:
            client._retry_with_backoff(fn)

        assert exc_info.value.__cause__ is original


# ── TestRetryIntegrationWithGenerate ─────────────────────────


class TestRetryIntegrationWithGenerate:
    """Tests that generate() and generate_stream() properly use retry logic."""

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_generate_retries_on_transient_error(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate() should retry and succeed after a transient failure."""
        mock_get_creds.return_value = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Success after retry"
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.total_token_count = 15
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        mock_response.candidates = [candidate]

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            google_exceptions.ServiceUnavailable("503"),
            mock_response,
        ]
        mock_model_cls.return_value = mock_model

        result = client.generate("Hello")

        assert result.text == "Success after retry"
        assert mock_model.generate_content.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_generate_raises_after_all_retries_exhausted(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.ResourceExhausted("429")
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiRateLimitError):
            client.generate("Hello")

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_generate_stream_retries_on_transient_error(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate_stream() materializes inside retry, so mid-stream errors retry."""
        mock_get_creds.return_value = MagicMock()

        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world"

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            google_exceptions.InternalServerError("500"),
            [chunk1, chunk2],  # Second attempt succeeds
        ]
        mock_model_cls.return_value = mock_model

        result = list(client.generate_stream("Say hello"))

        assert result == ["Hello ", "world"]
        assert mock_model.generate_content.call_count == 2

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_generate_stream_raises_after_all_retries_exhausted(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.ServiceUnavailable("503")
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiConnectionError):
            list(client.generate_stream("Hello"))

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_generate_non_retryable_error_not_retried(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        client: GeminiClient,
    ) -> None:
        """A non-retryable error like NotFound should propagate immediately."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.NotFound("model not found")
        mock_model_cls.return_value = mock_model

        with pytest.raises(google_exceptions.NotFound):
            client.generate("Hello")

        mock_model.generate_content.assert_called_once()
        mock_sleep.assert_not_called()


# ── TestIsSSLOrConnectionError ───────────────────────────────


class TestIsSSLOrConnectionError:
    """Tests for the _is_ssl_or_connection_error helper function."""

    def test_direct_ssl_error(self) -> None:
        assert _is_ssl_or_connection_error(ssl.SSLError("SSL handshake failed")) is True

    def test_direct_ssl_eof_error(self) -> None:
        assert _is_ssl_or_connection_error(ssl.SSLEOFError("EOF occurred")) is True

    def test_direct_connection_reset_error(self) -> None:
        assert _is_ssl_or_connection_error(ConnectionResetError("reset")) is True

    def test_direct_connection_aborted_error(self) -> None:
        assert _is_ssl_or_connection_error(ConnectionAbortedError("aborted")) is True

    def test_ssl_keyword_in_message(self) -> None:
        exc = OSError("SSL: CERTIFICATE_VERIFY_FAILED")
        assert _is_ssl_or_connection_error(exc) is True

    def test_eof_keyword_in_message(self) -> None:
        exc = OSError("EOF occurred in violation of protocol")
        assert _is_ssl_or_connection_error(exc) is True

    def test_certificate_keyword_in_message(self) -> None:
        exc = RuntimeError("certificate verify failed")
        assert _is_ssl_or_connection_error(exc) is True

    def test_handshake_keyword_in_message(self) -> None:
        exc = RuntimeError("handshake operation timed out")
        assert _is_ssl_or_connection_error(exc) is True

    def test_nested_ssl_error_via_cause(self) -> None:
        """An SSL error wrapped via __cause__ (raise ... from ...)."""
        inner = ssl.SSLEOFError("EOF occurred")
        outer = google_exceptions.ServiceUnavailable("503")
        outer.__cause__ = inner
        assert _is_ssl_or_connection_error(outer) is True

    def test_nested_ssl_error_via_context(self) -> None:
        """An SSL error wrapped via __context__ (implicit chaining)."""
        inner = ssl.SSLError("SSL error")
        outer = RuntimeError("transport failed")
        outer.__context__ = inner
        assert _is_ssl_or_connection_error(outer) is True

    def test_deeply_nested_ssl_error(self) -> None:
        """SSL error nested 3 levels deep should still be detected."""
        innermost = ssl.SSLEOFError("EOF occurred")
        mid = ConnectionError("connection failed")
        mid.__cause__ = innermost
        outer = google_exceptions.ServiceUnavailable("503")
        outer.__cause__ = mid
        assert _is_ssl_or_connection_error(outer) is True

    def test_non_ssl_error_returns_false(self) -> None:
        assert _is_ssl_or_connection_error(ValueError("bad value")) is False

    def test_not_found_returns_false(self) -> None:
        assert _is_ssl_or_connection_error(google_exceptions.NotFound("404")) is False

    def test_generic_os_error_returns_false(self) -> None:
        assert _is_ssl_or_connection_error(OSError("disk full")) is False

    def test_none_cause_does_not_crash(self) -> None:
        """An exception with no chained exceptions should return False for non-SSL."""
        exc = RuntimeError("something")
        assert exc.__cause__ is None
        assert _is_ssl_or_connection_error(exc) is False


# ── TestGCPConfigFallbackLocation ────────────────────────────


class TestGCPConfigFallbackLocation:
    """Tests for the fallback_location field in GCPConfig."""

    def test_default_fallback_location(self) -> None:
        cfg = GCPConfig(project_id="proj")
        assert cfg.fallback_location == "us-central1"

    def test_custom_fallback_location(self) -> None:
        cfg = GCPConfig(project_id="proj", fallback_location="europe-west1")
        assert cfg.fallback_location == "europe-west1"

    def test_empty_fallback_disables_fallback(self) -> None:
        cfg = GCPConfig(project_id="proj", fallback_location="")
        assert cfg.fallback_location == ""


# ── TestReinitializeWithFallback ─────────────────────────────


class TestReinitializeWithFallback:
    """Tests for GeminiClient._reinitialize_with_fallback()."""

    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_switches_to_fallback_location(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
    ) -> None:
        """After calling _reinitialize_with_fallback, _active_location should change."""
        mock_get_creds.return_value = MagicMock()
        settings = Settings(
            gcp=GCPConfig(project_id="proj", location="global", fallback_location="us-central1"),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        client.initialize()  # primary init

        assert client._active_location == "global"
        assert client._using_fallback is False

        client._reinitialize_with_fallback()

        assert client._active_location == "us-central1"
        assert client._using_fallback is True

    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_clears_model_cache_on_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        settings = Settings(
            gcp=GCPConfig(project_id="proj", location="global", fallback_location="us-central1"),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        client.initialize()
        client._models["gemini-2.5-pro"] = MagicMock()  # simulate cached model

        client._reinitialize_with_fallback()

        assert client._models == {}

    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_no_op_if_already_on_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
    ) -> None:
        """If already using fallback, _reinitialize_with_fallback is a no-op."""
        mock_get_creds.return_value = MagicMock()
        settings = Settings(
            gcp=GCPConfig(project_id="proj", location="global", fallback_location="us-central1"),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        client.initialize()
        client._reinitialize_with_fallback()  # first fallback

        init_count = mock_vertexai_init.call_count

        client._reinitialize_with_fallback()  # second call — should be no-op

        assert mock_vertexai_init.call_count == init_count  # no additional init
        assert client._active_location == "us-central1"

    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_no_op_if_fallback_same_as_primary(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
    ) -> None:
        """If fallback == primary location, no reinit happens."""
        mock_get_creds.return_value = MagicMock()
        settings = Settings(
            gcp=GCPConfig(project_id="proj", location="us-central1", fallback_location="us-central1"),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        client.initialize()

        init_count = mock_vertexai_init.call_count

        client._reinitialize_with_fallback()

        assert mock_vertexai_init.call_count == init_count
        assert client._using_fallback is False

    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_no_op_if_fallback_empty(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
    ) -> None:
        """If fallback_location is empty, no reinit happens."""
        mock_get_creds.return_value = MagicMock()
        settings = Settings(
            gcp=GCPConfig(project_id="proj", location="global", fallback_location=""),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        client.initialize()

        init_count = mock_vertexai_init.call_count

        client._reinitialize_with_fallback()

        assert mock_vertexai_init.call_count == init_count
        assert client._using_fallback is False


# ── TestLocationFallbackInRetry ──────────────────────────────


class TestLocationFallbackInRetry:
    """Tests for SSL fallback behavior inside _retry_with_backoff."""

    @pytest.fixture()
    def fallback_settings(self) -> Settings:
        """Settings with global as primary and us-central1 as fallback."""
        return Settings(
            gcp=GCPConfig(project_id="test-project", location="global", fallback_location="us-central1"),
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
    def fallback_client(self, fallback_settings: Settings) -> GeminiClient:
        return GeminiClient(fallback_settings)

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_ssl_error_triggers_fallback_and_succeeds(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """An SSL error should trigger fallback, and the retry should succeed."""
        mock_get_creds.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from fallback"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15,
        )
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        mock_model = MagicMock()
        # First call fails with SSL, second call (after fallback) succeeds
        mock_model.generate_content.side_effect = [
            ssl.SSLEOFError("EOF occurred in violation of protocol"),
            mock_response,
        ]
        mock_model_cls.return_value = mock_model

        result = fallback_client.generate("Hello")

        assert result.text == "Hello from fallback"
        assert fallback_client._using_fallback is True
        assert fallback_client._active_location == "us-central1"

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_ssl_error_fallback_also_fails_raises_connection_error(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """If both primary and fallback fail, raises GeminiConnectionError."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = ssl.SSLEOFError("EOF occurred")
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiConnectionError):
            fallback_client.generate("Hello")

        assert fallback_client._using_fallback is True

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_nested_ssl_error_triggers_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A nested SSL error (wrapped in ServiceUnavailable) should trigger fallback."""
        mock_get_creds.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5, candidates_token_count=10, total_token_count=15,
        )
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        inner_ssl = ssl.SSLEOFError("EOF occurred")
        outer_exc = google_exceptions.ServiceUnavailable("503 SSL issue")
        outer_exc.__cause__ = inner_ssl

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [outer_exc, mock_response]
        mock_model_cls.return_value = mock_model

        # ServiceUnavailable IS in _RETRYABLE_EXCEPTIONS, so it will be retried
        # normally (not via fallback). But if ALL retries exhaust and the inner
        # cause is SSL, it should still be caught. Let's test a pure wrapped case:
        # Use an exception that is NOT retryable but wraps SSL.
        non_retryable_wrapper = RuntimeError("transport error")
        non_retryable_wrapper.__cause__ = ssl.SSLEOFError("EOF occurred")

        mock_model.generate_content.side_effect = [non_retryable_wrapper, mock_response]

        result = fallback_client.generate("Hello")
        assert result.text == "OK"
        assert fallback_client._using_fallback is True

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_non_ssl_error_does_not_trigger_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A non-SSL, non-retryable error should propagate immediately, no fallback."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.NotFound("model not found")
        mock_model_cls.return_value = mock_model

        with pytest.raises(google_exceptions.NotFound):
            fallback_client.generate("Hello")

        assert fallback_client._using_fallback is False
        assert fallback_client._active_location == "global"
        mock_model.generate_content.assert_called_once()

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_retryable_error_does_not_trigger_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A plain 503 (no SSL) should be retried normally, NOT trigger fallback."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.ServiceUnavailable("503")
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiConnectionError):
            fallback_client.generate("Hello")

        assert fallback_client._using_fallback is False
        assert fallback_client._active_location == "global"
        # Should have retried max_retries + 1 times (4 total)
        assert mock_model.generate_content.call_count == 4

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_client_init_tracks_primary_location(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """Verify the client starts with the correct primary location state."""
        assert fallback_client._active_location == "global"
        assert fallback_client._using_fallback is False


# ── TestProxyResponseParseError ──────────────────────────────


class TestProxyResponseParseError:
    """Tests for detecting VPN/proxy malformed response errors.

    When a VPN/proxy intercepts a request and returns a malformed error body
    (e.g. a JSON array instead of a dict), the google-api-core SDK crashes in
    ``format_http_response_error`` with:
        ``AttributeError: 'list' object has no attribute 'get'``

    These tests verify that ``_is_ssl_or_connection_error`` detects this
    pattern and that the retry loop triggers location fallback.
    """

    # ── Detection tests ──────────────────────────────────────

    def test_attribute_error_get_detected(self) -> None:
        """Direct AttributeError about .get() should be detected as proxy error."""
        exc = AttributeError("'list' object has no attribute 'get'")
        assert _is_ssl_or_connection_error(exc) is True

    def test_attribute_error_different_attr_not_detected(self) -> None:
        """AttributeError about a different attribute should NOT be detected."""
        exc = AttributeError("'NoneType' object has no attribute 'text'")
        assert _is_ssl_or_connection_error(exc) is False

    def test_attribute_error_get_nested_in_service_unavailable(self) -> None:
        """AttributeError wrapped inside ServiceUnavailable should be detected."""
        inner = AttributeError("'list' object has no attribute 'get'")
        outer = google_exceptions.ServiceUnavailable("503")
        outer.__cause__ = inner
        assert _is_ssl_or_connection_error(outer) is True

    def test_attribute_error_get_nested_in_internal_error(self) -> None:
        """AttributeError wrapped inside InternalServerError should be detected."""
        inner = AttributeError("'list' object has no attribute 'get'")
        outer = google_exceptions.InternalServerError("500")
        outer.__cause__ = inner
        assert _is_ssl_or_connection_error(outer) is True

    def test_attribute_error_get_via_context_chain(self) -> None:
        """AttributeError via __context__ (implicit chaining) should be detected."""
        inner = AttributeError("'list' object has no attribute 'get'")
        outer = RuntimeError("failed to format error")
        outer.__context__ = inner
        assert _is_ssl_or_connection_error(outer) is True

    # ── Fallback integration tests ───────────────────────────

    @pytest.fixture()
    def fallback_settings(self) -> Settings:
        """Settings with global as primary and us-central1 as fallback."""
        return Settings(
            gcp=GCPConfig(project_id="test-project", location="global", fallback_location="us-central1"),
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
    def fallback_client(self, fallback_settings: Settings) -> GeminiClient:
        return GeminiClient(fallback_settings)

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_direct_attribute_error_triggers_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A direct AttributeError about .get() should trigger fallback and succeed."""
        mock_get_creds.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from fallback"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15,
        )
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            AttributeError("'list' object has no attribute 'get'"),
            mock_response,
        ]
        mock_model_cls.return_value = mock_model

        result = fallback_client.generate("Hello")

        assert result.text == "Hello from fallback"
        assert fallback_client._using_fallback is True
        assert fallback_client._active_location == "us-central1"

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_retryable_wrapping_proxy_error_triggers_fallback(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A ServiceUnavailable wrapping the .get() AttributeError should trigger fallback
        on the first attempt instead of burning all retries."""
        mock_get_creds.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Fallback success"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15,
        )
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        # ServiceUnavailable wrapping the AttributeError from format_http_response_error
        inner = AttributeError("'list' object has no attribute 'get'")
        outer = google_exceptions.ServiceUnavailable("503")
        outer.__cause__ = inner

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [outer, mock_response]
        mock_model_cls.return_value = mock_model

        result = fallback_client.generate("Hello")

        assert result.text == "Fallback success"
        assert fallback_client._using_fallback is True
        assert fallback_client._active_location == "us-central1"
        # Should NOT have retried 4 times — should have fallback on first attempt
        assert mock_model.generate_content.call_count == 2

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_proxy_error_fallback_also_fails_raises_connection_error(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """If both primary (proxy error) and fallback fail, raises GeminiConnectionError."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = AttributeError(
            "'list' object has no attribute 'get'"
        )
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiConnectionError):
            fallback_client.generate("Hello")

        assert fallback_client._using_fallback is True

    @patch("vaig.core.client.time.sleep")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_plain_retryable_503_still_retries_normally(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_sleep: MagicMock,
        fallback_client: GeminiClient,
    ) -> None:
        """A plain 503 (no proxy/SSL cause) should still retry normally, NOT fallback."""
        mock_get_creds.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = google_exceptions.ServiceUnavailable("503")
        mock_model_cls.return_value = mock_model

        with pytest.raises(GeminiConnectionError):
            fallback_client.generate("Hello")

        assert fallback_client._using_fallback is False
        assert fallback_client._active_location == "global"
        # Should have retried max_retries + 1 times (4 total)
        assert mock_model.generate_content.call_count == 4

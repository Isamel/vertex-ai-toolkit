"""Tests for retry logic, custom exceptions, and RetryConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from vaig.core.client import GeminiClient, _RETRYABLE_EXCEPTIONS
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

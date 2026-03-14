"""Tests for gcloud CLI token auto-refresh logic.

Covers:
- _fetch_gcloud_access_token() — subprocess wrapper
- _gcloud_refresh_handler() — refresh callback for google-auth
- _get_gcloud_token_credentials() — credential factory with refresh_handler
- Integration: token expiry detection and transparent refresh
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from google.auth import exceptions as auth_exceptions
from google.oauth2.credentials import Credentials as OAuth2Credentials


# ══════════════════════════════════════════════════════════════
# _fetch_gcloud_access_token
# ══════════════════════════════════════════════════════════════


class TestFetchGcloudAccessToken:
    """Test the low-level gcloud CLI subprocess wrapper."""

    @patch("vaig.core.auth.subprocess.run")
    def test_returns_token_on_success(self, mock_run: MagicMock) -> None:
        """Should return the stripped stdout when gcloud succeeds."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout="ya29.test-token-value\n",
            stderr="",
            returncode=0,
        )

        token = _fetch_gcloud_access_token()

        assert token == "ya29.test-token-value"
        mock_run.assert_called_once_with(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )

    @patch("vaig.core.auth.subprocess.run")
    def test_raises_on_empty_token(self, mock_run: MagicMock) -> None:
        """Should raise RuntimeError when gcloud returns empty output."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout="",
            stderr="",
            returncode=0,
        )

        with pytest.raises(RuntimeError, match="Could not obtain credentials"):
            _fetch_gcloud_access_token()

    @patch("vaig.core.auth.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        """Should raise RuntimeError when gcloud exits with non-zero code."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout="",
            stderr="ERROR: not authenticated",
            returncode=1,
        )

        with pytest.raises(RuntimeError, match="Could not obtain credentials"):
            _fetch_gcloud_access_token()

    @patch("vaig.core.auth.subprocess.run")
    def test_includes_stderr_in_error(self, mock_run: MagicMock) -> None:
        """Error message should include gcloud stderr for debugging."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout="",
            stderr="ERROR: gcloud not configured",
            returncode=1,
        )

        with pytest.raises(RuntimeError, match="gcloud not configured"):
            _fetch_gcloud_access_token()

    @patch("vaig.core.auth.subprocess.run", side_effect=FileNotFoundError)
    def test_raises_on_gcloud_not_found(self, mock_run: MagicMock) -> None:
        """Should raise RuntimeError when gcloud CLI is not installed."""
        from vaig.core.auth import _fetch_gcloud_access_token

        with pytest.raises(RuntimeError, match="gcloud CLI not found"):
            _fetch_gcloud_access_token()

    @patch(
        "vaig.core.auth.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="gcloud", timeout=10),
    )
    def test_raises_on_timeout(self, mock_run: MagicMock) -> None:
        """Should raise RuntimeError when gcloud command times out."""
        from vaig.core.auth import _fetch_gcloud_access_token

        with pytest.raises(RuntimeError, match="timed out"):
            _fetch_gcloud_access_token()


# ══════════════════════════════════════════════════════════════
# _gcloud_refresh_handler
# ══════════════════════════════════════════════════════════════


class TestGcloudRefreshHandler:
    """Test the refresh_handler callback for google-auth."""

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_returns_token_and_expiry(self, mock_fetch: MagicMock) -> None:
        """Should return (token, naive_utc_expiry) tuple on success."""
        from vaig.core.auth import _GCLOUD_TOKEN_LIFETIME, _gcloud_refresh_handler

        mock_fetch.return_value = "ya29.refreshed-token"

        before = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        token, expiry = _gcloud_refresh_handler(request=None)
        after = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        assert token == "ya29.refreshed-token"
        assert isinstance(expiry, datetime)
        # google-auth expects naive UTC datetimes
        assert expiry.tzinfo is None
        # Expiry should be ~1h from now
        assert before + _GCLOUD_TOKEN_LIFETIME <= expiry <= after + _GCLOUD_TOKEN_LIFETIME

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_accepts_scopes_kwarg(self, mock_fetch: MagicMock) -> None:
        """refresh_handler must accept scopes kwarg (required by google-auth API)."""
        from vaig.core.auth import _gcloud_refresh_handler

        mock_fetch.return_value = "ya29.token"

        # Should not raise — scopes is accepted but unused
        token, expiry = _gcloud_refresh_handler(request=None, scopes=["scope1"])
        assert token == "ya29.token"

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_raises_refresh_error_on_failure(self, mock_fetch: MagicMock) -> None:
        """Should wrap RuntimeError into google.auth RefreshError."""
        from vaig.core.auth import _gcloud_refresh_handler

        mock_fetch.side_effect = RuntimeError("gcloud CLI not found")

        with pytest.raises(auth_exceptions.RefreshError, match="re-authenticate"):
            _gcloud_refresh_handler(request=None)

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_refresh_error_chains_cause(self, mock_fetch: MagicMock) -> None:
        """RefreshError should chain the original RuntimeError as __cause__."""
        from vaig.core.auth import _gcloud_refresh_handler

        original = RuntimeError("original error")
        mock_fetch.side_effect = original

        with pytest.raises(auth_exceptions.RefreshError) as exc_info:
            _gcloud_refresh_handler(request=None)

        assert exc_info.value.__cause__ is original


# ══════════════════════════════════════════════════════════════
# _get_gcloud_token_credentials
# ══════════════════════════════════════════════════════════════


class TestGetGcloudTokenCredentials:
    """Test the credential factory with refresh_handler."""

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_returns_oauth2_credentials(self, mock_fetch: MagicMock) -> None:
        """Should return OAuth2Credentials instance."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.return_value = "ya29.initial-token"

        creds = _get_gcloud_token_credentials()

        assert isinstance(creds, OAuth2Credentials)
        assert creds.token == "ya29.initial-token"

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_has_expiry_set(self, mock_fetch: MagicMock) -> None:
        """Credentials should have an expiry time set (~1h from now)."""
        from vaig.core.auth import _GCLOUD_TOKEN_LIFETIME, _get_gcloud_token_credentials

        mock_fetch.return_value = "ya29.initial-token"

        before = datetime.now(tz=timezone.utc)
        creds = _get_gcloud_token_credentials()
        after = datetime.now(tz=timezone.utc)

        assert creds.expiry is not None
        # google-auth stores expiry as naive UTC — compare accordingly
        before_naive = before.replace(tzinfo=None)
        after_naive = after.replace(tzinfo=None)
        assert before_naive + _GCLOUD_TOKEN_LIFETIME <= creds.expiry <= after_naive + _GCLOUD_TOKEN_LIFETIME

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_has_refresh_handler(self, mock_fetch: MagicMock) -> None:
        """Credentials should have a refresh_handler registered."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.return_value = "ya29.initial-token"

        creds = _get_gcloud_token_credentials()

        assert creds.refresh_handler is not None

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_propagates_fetch_error(self, mock_fetch: MagicMock) -> None:
        """Should propagate RuntimeError from _fetch_gcloud_access_token."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.side_effect = RuntimeError("gcloud CLI not found")

        with pytest.raises(RuntimeError, match="gcloud CLI not found"):
            _get_gcloud_token_credentials()


# ══════════════════════════════════════════════════════════════
# Integration: token refresh via google-auth machinery
# ══════════════════════════════════════════════════════════════


class TestTokenRefreshIntegration:
    """Test that the refresh_handler integrates correctly with google-auth."""

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_refresh_updates_token(self, mock_fetch: MagicMock) -> None:
        """Calling refresh() should update the token via the handler."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.side_effect = ["ya29.initial", "ya29.refreshed"]

        creds = _get_gcloud_token_credentials()
        assert creds.token == "ya29.initial"

        # Force expiry so refresh() is triggered (naive UTC as google-auth expects)
        creds.expiry = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)

        # Refresh should fetch a new token
        mock_request = MagicMock()
        creds.refresh(mock_request)

        assert creds.token == "ya29.refreshed"
        assert creds.expiry is not None

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_before_request_refreshes_expired_token(self, mock_fetch: MagicMock) -> None:
        """before_request() should auto-refresh an expired token."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.side_effect = ["ya29.initial", "ya29.auto-refreshed"]

        creds = _get_gcloud_token_credentials()
        assert creds.token == "ya29.initial"

        # Expire the token (naive UTC)
        creds.expiry = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)

        # before_request should detect expiry and refresh
        mock_request = MagicMock()
        headers: dict[str, str] = {}
        creds.before_request(mock_request, "GET", "https://example.com", headers)

        assert creds.token == "ya29.auto-refreshed"
        # Authorization header should contain the new token
        assert "ya29.auto-refreshed" in headers.get("authorization", "")

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_before_request_skips_refresh_for_valid_token(self, mock_fetch: MagicMock) -> None:
        """before_request() should NOT refresh when token is still valid."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.return_value = "ya29.still-valid"

        creds = _get_gcloud_token_credentials()

        # Token is valid — should NOT trigger another fetch
        mock_request = MagicMock()
        headers: dict[str, str] = {}
        creds.before_request(mock_request, "GET", "https://example.com", headers)

        # _fetch should only have been called once (initial)
        assert mock_fetch.call_count == 1
        assert "ya29.still-valid" in headers.get("authorization", "")

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    def test_refresh_error_does_not_crash(self, mock_fetch: MagicMock) -> None:
        """Refresh failure should raise RefreshError, not crash."""
        from vaig.core.auth import _get_gcloud_token_credentials

        mock_fetch.side_effect = [
            "ya29.initial",
            RuntimeError("gcloud session expired"),
        ]

        creds = _get_gcloud_token_credentials()

        # Expire the token (naive UTC)
        creds.expiry = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)

        # Refresh should raise RefreshError (not RuntimeError)
        mock_request = MagicMock()
        with pytest.raises(auth_exceptions.RefreshError, match="re-authenticate"):
            creds.refresh(mock_request)


# ══════════════════════════════════════════════════════════════
# ADC fallback integration
# ══════════════════════════════════════════════════════════════


class TestADCFallbackToGcloudToken:
    """Test that _get_adc_credentials correctly falls back to refreshable gcloud token."""

    @patch("vaig.core.auth._fetch_gcloud_access_token")
    @patch("vaig.core.auth.google.auth.default")
    def test_fallback_creates_refreshable_credentials(
        self,
        mock_adc: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        """When ADC fails, fallback should create credentials with refresh_handler."""
        from vaig.core.auth import _get_adc_credentials

        mock_adc.side_effect = google_auth_default_error()
        mock_fetch.return_value = "ya29.fallback-token"

        creds = _get_adc_credentials()

        assert isinstance(creds, OAuth2Credentials)
        assert creds.token == "ya29.fallback-token"
        assert creds.refresh_handler is not None
        assert creds.expiry is not None


def google_auth_default_error() -> Exception:
    """Create a DefaultCredentialsError for testing."""
    import google.auth.exceptions

    return google.auth.exceptions.DefaultCredentialsError("No ADC configured")

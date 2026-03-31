"""Tests for FastAPI dependencies — Task 2.2.

Covers:
- get_current_user: IAP header extraction, prefix stripping, env fallback
- get_settings: form/query params → Settings.from_overrides()
- get_container: build_container wrapper
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from fastapi import HTTPException

from vaig.web.deps import get_container, get_current_user, get_settings

# ── get_current_user ─────────────────────────────────────────


class TestGetCurrentUser:
    """Tests for IAP header extraction and dev fallback."""

    def _make_request(self, headers: dict[str, str] | None = None) -> MagicMock:
        """Create a mock request with given headers."""
        request = MagicMock()
        request.headers = headers or {}
        return request

    def test_extracts_email_from_iap_header(self) -> None:
        """Should extract email from X-Goog-Authenticated-User-Email."""
        request = self._make_request(
            {"X-Goog-Authenticated-User-Email": "accounts.google.com:user@example.com"}
        )
        assert get_current_user(request) == "user@example.com"

    def test_strips_iap_prefix(self) -> None:
        """Should strip the accounts.google.com: prefix."""
        request = self._make_request(
            {"X-Goog-Authenticated-User-Email": "accounts.google.com:admin@corp.com"}
        )
        assert get_current_user(request) == "admin@corp.com"

    def test_returns_raw_header_without_prefix(self) -> None:
        """If header doesn't have the IAP prefix, return as-is."""
        request = self._make_request(
            {"X-Goog-Authenticated-User-Email": "user@direct.com"}
        )
        assert get_current_user(request) == "user@direct.com"

    @patch.dict(
        "os.environ",
        {"VAIG_WEB_DEV_USER": "test-dev@local", "VAIG_WEB_DEV_MODE": "true"},
    )
    def test_falls_back_to_env_var(self) -> None:
        """Should use VAIG_WEB_DEV_USER when no IAP header and dev mode on."""
        request = self._make_request({})
        assert get_current_user(request) == "test-dev@local"

    @patch.dict("os.environ", {"VAIG_WEB_DEV_MODE": "true"}, clear=True)
    def test_falls_back_to_default(self) -> None:
        """Should use dev@localhost when no header, no env var, but dev mode on."""
        import os

        os.environ.pop("VAIG_WEB_DEV_USER", None)
        request = self._make_request({})
        assert get_current_user(request) == "dev@localhost"

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_401_without_dev_mode(self) -> None:
        """Should raise HTTP 401 when no IAP header and dev mode is off."""
        import os

        os.environ.pop("VAIG_WEB_DEV_MODE", None)
        request = self._make_request({})
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(request)
        assert exc_info.value.status_code == 401


# ── get_settings ─────────────────────────────────────────────


class TestGetSettings:
    """Tests for form/query param → Settings.from_overrides()."""

    @pytest.mark.asyncio
    async def test_reads_form_data_for_post(self) -> None:
        """POST with form data should pass overrides to from_overrides()."""
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.query_params = {}
        form_data = {"project": "test-proj", "model": "gemini-2.0-flash"}
        request.form = AsyncMock(return_value=form_data)

        with patch("vaig.web.deps.Settings.from_overrides") as mock_from:
            mock_from.return_value = MagicMock()
            await get_settings(request)
            mock_from.assert_called_once_with(project="test-proj", model="gemini-2.0-flash")

    @pytest.mark.asyncio
    async def test_reads_query_params_for_get(self) -> None:
        """GET with query params should pass overrides to from_overrides()."""
        request = MagicMock()
        request.headers = {"content-type": "text/html"}
        request.query_params = {"project": "query-proj"}

        with patch("vaig.web.deps.Settings.from_overrides") as mock_from:
            mock_from.return_value = MagicMock()
            await get_settings(request)
            mock_from.assert_called_once_with(project="query-proj")

    @pytest.mark.asyncio
    async def test_skips_empty_values(self) -> None:
        """Empty form values should not be passed as overrides."""
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.query_params = {}
        form_data = {"project": "", "model": "  ", "temperature": "0.5"}
        request.form = AsyncMock(return_value=form_data)

        with patch("vaig.web.deps.Settings.from_overrides") as mock_from:
            mock_from.return_value = MagicMock()
            await get_settings(request)
            mock_from.assert_called_once_with(temperature=0.5)

    @pytest.mark.asyncio
    async def test_converts_temperature_to_float(self) -> None:
        """Temperature string should be converted to float."""
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.query_params = {}
        form_data = {"temperature": "0.7"}
        request.form = AsyncMock(return_value=form_data)

        with patch("vaig.web.deps.Settings.from_overrides") as mock_from:
            mock_from.return_value = MagicMock()
            await get_settings(request)
            mock_from.assert_called_once_with(temperature=0.7)

    @pytest.mark.asyncio
    async def test_invalid_temperature_skipped(self) -> None:
        """Non-numeric temperature should be silently skipped."""
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.query_params = {}
        form_data = {"temperature": "not-a-number", "project": "valid-proj"}
        request.form = AsyncMock(return_value=form_data)

        with patch("vaig.web.deps.Settings.from_overrides") as mock_from:
            mock_from.return_value = MagicMock()
            await get_settings(request)
            mock_from.assert_called_once_with(project="valid-proj")


# ── get_container ────────────────────────────────────────────


class TestGetContainer:
    """Tests for the container dependency."""

    def test_calls_build_container(self) -> None:
        """get_container should delegate to build_container()."""
        mock_settings = MagicMock()
        with patch("vaig.web.deps.build_container") as mock_build:
            mock_build.return_value = MagicMock()
            result = get_container(mock_settings)
            mock_build.assert_called_once_with(mock_settings)
            assert result is mock_build.return_value

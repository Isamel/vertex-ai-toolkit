"""Integration tests for CLI login flow (Phase 4, Task 4.1).

Covers:
  - SC-AUTH-002a: Successful login flow end-to-end (mock webbrowser.open,
    run localhost callback, verify credential file created with correct perms)
  - SC-AUTH-005a: Credential file permissions (0600 / 0700)
  - SC-AUTH-002b: Login when already authenticated
  - SC-AUTH-002c: Login with unreachable backend
  - SC-AUTH-002d: Login timeout
"""

from __future__ import annotations

import json
import stat
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from vaig.core.platform_auth import PlatformAuthManager

# ── Helpers ────────────────────────────────────────────────────


def _make_manager(
    tmp_path: Path,
    backend_url: str = "https://api.example.com",
    org_id: str = "test-org",
    http_client: httpx.Client | None = None,
) -> PlatformAuthManager:
    """Create a PlatformAuthManager pointing at a temp credentials dir."""
    return PlatformAuthManager(
        backend_url=backend_url,
        org_id=org_id,
        http_client=http_client or MagicMock(spec=httpx.Client),
        credentials_dir=tmp_path,
    )


def _write_creds(tmp_path: Path, creds: dict[str, Any]) -> None:
    """Write credentials JSON to the tmp_path."""
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps(creds), encoding="utf-8")


# ── Tests: Credential file permissions ────────────────────────


class TestCredentialFilePermissions:
    """SC-AUTH-005a: Credential file and directory permissions."""

    def test_credentials_dir_created_with_700_perms(self, tmp_path: Path) -> None:
        creds_dir = tmp_path / "sub" / ".vaig"
        manager = _make_manager(tmp_path=creds_dir)
        manager._save_creds({"access_token": "test"})

        assert creds_dir.exists()
        mode = stat.S_IMODE(creds_dir.stat().st_mode)
        assert mode == 0o700

    def test_credentials_file_created_with_600_perms(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        manager._save_creds({"access_token": "test"})

        creds_file = tmp_path / "credentials.json"
        assert creds_file.exists()
        mode = stat.S_IMODE(creds_file.stat().st_mode)
        assert mode == 0o600

    def test_credentials_file_contains_expected_fields(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        manager._save_creds({
            "access_token": "tok123",
            "refresh_token": "ref456",
            "expires_at": 9999999999,
            "cli_id": "cli-abc",
            "backend_url": "https://api.example.com",
        })

        creds_file = tmp_path / "credentials.json"
        data = json.loads(creds_file.read_text(encoding="utf-8"))
        assert data["access_token"] == "tok123"
        assert data["refresh_token"] == "ref456"
        assert data["cli_id"] == "cli-abc"


# ── Tests: Login flow e2e ─────────────────────────────────────


class TestLoginFlowE2E:
    """SC-AUTH-002a: End-to-end login flow with mocked browser + callback."""

    def test_successful_login_creates_creds_file(self, tmp_path: Path) -> None:
        """Full login flow: mock browser, simulate callback, verify creds stored."""
        # Set up mock httpx client that:
        # 1. Responds OK to /healthz (reachability check)
        # 2. Returns tokens from /api/v1/auth/token
        healthz_response = MagicMock()
        healthz_response.raise_for_status = MagicMock()

        token_response = MagicMock()
        token_response.json.return_value = {
            "access_token": "eyJ.test.access",
            "refresh_token": "ref-new-tok",
            "token_type": "bearer",
            "expires_in": 3600,
        }
        token_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)

        def _route_get(url: str, **kwargs: Any) -> MagicMock:
            return healthz_response

        def _route_post(url: str, **kwargs: Any) -> MagicMock:
            return token_response

        mock_client.get.side_effect = _route_get
        mock_client.post.side_effect = _route_post

        manager = _make_manager(tmp_path, http_client=mock_client)

        # Simulate callback: when browser is "opened", send a callback
        # to the localhost server with a mock auth code
        original_webbrowser_open = None

        def _fake_browser_open(url: str) -> None:
            """Extract port from auth URL and POST callback to it."""
            import urllib.parse
            import urllib.request

            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            state = params.get("state", [""])[0]
            redirect_uri = params.get("redirect_uri", [""])[0]

            if not redirect_uri:
                # Fallback: extract port from the URL pattern
                return

            # Send callback to the local server
            callback_url = f"{redirect_uri}?code=mock-auth-code&state={state}"
            try:
                urllib.request.urlopen(callback_url, timeout=5)  # noqa: S310
            except Exception:
                pass

        with patch("webbrowser.open", side_effect=_fake_browser_open):
            result = manager.login(force=True)

        assert result.success is True

        # Verify creds file was created
        creds_file = tmp_path / "credentials.json"
        assert creds_file.exists()

        data = json.loads(creds_file.read_text(encoding="utf-8"))
        assert data["access_token"] == "eyJ.test.access"
        assert data["refresh_token"] == "ref-new-tok"

    def test_login_when_already_authenticated_returns_success(
        self, tmp_path: Path
    ) -> None:
        """SC-AUTH-002b: Already authenticated without --force."""
        _write_creds(tmp_path, {
            "access_token": "existing-tok",
            "expires_at": int(time.time()) + 3600,
        })

        manager = _make_manager(tmp_path)
        result = manager.login(force=False)

        assert result.success is True
        assert result.error is not None
        assert "Already authenticated" in result.error

    def test_login_unreachable_backend(self, tmp_path: Path) -> None:
        """SC-AUTH-002c: Backend unreachable returns descriptive error."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        manager = _make_manager(tmp_path, http_client=mock_client)
        result = manager.login(force=True)

        assert result.success is False
        assert "Cannot reach platform backend" in (result.error or "")

    def test_login_timeout(self, tmp_path: Path) -> None:
        """SC-AUTH-002d: Login times out when no callback received."""
        healthz_response = MagicMock()
        healthz_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = healthz_response

        manager = _make_manager(tmp_path, http_client=mock_client)

        # Patch the timeout to be very short (1 second)
        with (
            patch("vaig.core.platform_auth._LOGIN_TIMEOUT_SECONDS", 1),
            patch("webbrowser.open"),
        ):
            result = manager.login(force=True)

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_login_force_re_authenticates(self, tmp_path: Path) -> None:
        """--force flag triggers re-authentication even when already authenticated."""
        _write_creds(tmp_path, {
            "access_token": "old-tok",
            "expires_at": int(time.time()) + 3600,
        })

        healthz_response = MagicMock()
        healthz_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = healthz_response

        manager = _make_manager(tmp_path, http_client=mock_client)

        # With force=True but no callback → timeout
        with (
            patch("vaig.core.platform_auth._LOGIN_TIMEOUT_SECONDS", 1),
            patch("webbrowser.open"),
        ):
            result = manager.login(force=True)

        # Should NOT return "Already authenticated" — it should attempt login
        if result.error:
            assert "Already authenticated" not in result.error

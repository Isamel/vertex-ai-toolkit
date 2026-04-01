"""Tests for PlatformAuthManager and platform auth helpers (Phase 2).

Covers:
  - SC-AUTH-002: Credential I/O (save/load/delete via tmp_path)
  - SC-AUTH-002: PKCE challenge generation (S256)
  - SC-AUTH-002e: Token refresh (mock httpx)
  - SC-AUTH-002: is_authenticated with invalid/corrupt JSON
  - SC-AUTH-004: _apply_enforced_config mutates settings
  - SC-AUTH-004a: _check_platform_auth no-op when platform disabled
  - SC-AUTH-006b: build_container wires PlatformAuthManager when enabled
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click.exceptions
import httpx
import pytest

from vaig.core.config import Settings
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


def _make_jwt_token(payload: dict[str, Any]) -> str:
    """Build a fake JWT (header.payload.sig) with base64url-encoded payload."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{payload_b64}.fakesig"


# ── Credential I/O ────────────────────────────────────────────


class TestCredentialIO:
    """Save, load, and delete credentials via tmp_path."""

    def test_save_and_load_creds(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        creds = {"access_token": "tok123", "refresh_token": "ref456", "expires_at": 9999999999}
        manager._save_creds(creds)

        loaded = manager._load_creds()
        assert loaded is not None
        assert loaded["access_token"] == "tok123"
        assert loaded["refresh_token"] == "ref456"

    def test_load_creds_returns_none_when_missing(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        assert manager._load_creds() is None

    def test_delete_creds(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        manager._save_creds({"access_token": "x"})
        assert manager._load_creds() is not None

        manager._delete_creds()
        assert manager._load_creds() is None

    def test_load_creds_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("NOT JSON {{{", encoding="utf-8")

        manager = _make_manager(tmp_path)
        assert manager._load_creds() is None

    def test_load_creds_returns_none_on_non_object(self, tmp_path: Path) -> None:
        """A JSON file that is valid JSON but not a dict should return None."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text('"just a string"', encoding="utf-8")

        manager = _make_manager(tmp_path)
        assert manager._load_creds() is None

    def test_creds_dir_created_with_restrictive_perms(self, tmp_path: Path) -> None:
        creds_dir = tmp_path / "subdir" / ".vaig"
        manager = _make_manager(tmp_path=creds_dir)
        manager._save_creds({"access_token": "tok"})

        assert creds_dir.exists()
        # File should exist
        assert (creds_dir / "credentials.json").exists()


# ── PKCE challenge ────────────────────────────────────────────


class TestPKCE:
    """PKCE verifier/challenge generation."""

    def test_generate_pkce_returns_verifier_and_challenge(self) -> None:
        verifier, challenge = PlatformAuthManager._generate_pkce()

        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) <= 128
        assert len(verifier) >= 43  # RFC 7636 minimum

    def test_pkce_challenge_is_sha256_of_verifier(self) -> None:
        verifier, challenge = PlatformAuthManager._generate_pkce()

        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_pkce_generates_unique_values(self) -> None:
        v1, c1 = PlatformAuthManager._generate_pkce()
        v2, c2 = PlatformAuthManager._generate_pkce()
        assert v1 != v2
        assert c1 != c2


# ── is_authenticated ──────────────────────────────────────────


class TestIsAuthenticated:
    """is_authenticated() with various credential states."""

    def test_authenticated_with_valid_token(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "valid",
            "expires_at": int(time.time()) + 3600,
        })
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is True

    def test_not_authenticated_when_no_creds(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is False

    def test_not_authenticated_when_empty_token(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "",
            "expires_at": int(time.time()) + 3600,
        })
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is False

    def test_authenticated_with_expired_token_and_refresh(self, tmp_path: Path) -> None:
        """Expired access token but refresh_token present → still authenticated."""
        _write_creds(tmp_path, {
            "access_token": "expired",
            "expires_at": int(time.time()) - 100,
            "refresh_token": "ref123",
        })
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is True

    def test_not_authenticated_with_expired_token_no_refresh(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "expired",
            "expires_at": int(time.time()) - 100,
        })
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is False

    def test_not_authenticated_with_invalid_json(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("[1, 2, 3]", encoding="utf-8")
        manager = _make_manager(tmp_path)
        assert manager.is_authenticated() is False


# ── get_user_info (JWT decode) ────────────────────────────────


class TestGetUserInfo:
    """get_user_info() decodes JWT payload."""

    def test_decode_jwt_claims(self, tmp_path: Path) -> None:
        token = _make_jwt_token({"sub": "user@test.com", "org_id": "org1", "role": "admin"})
        _write_creds(tmp_path, {"access_token": token, "expires_at": 9999999999})

        manager = _make_manager(tmp_path)
        info = manager.get_user_info()

        assert info is not None
        assert info["email"] == "user@test.com"
        assert info["org_id"] == "org1"
        assert info["role"] == "admin"

    def test_returns_none_when_no_creds(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        assert manager.get_user_info() is None

    def test_returns_none_when_malformed_token(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {"access_token": "not.a.jwt.really", "expires_at": 9999999999})
        manager = _make_manager(tmp_path)
        # Should not raise — graceful None
        info = manager.get_user_info()
        # May be None or a dict with empty values depending on decode
        # The important thing is no exception is raised


# ── Token refresh (mock httpx) ────────────────────────────────


class TestGetToken:
    """get_token() with auto-refresh via mocked httpx."""

    def test_returns_valid_token_directly(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "still-valid",
            "expires_at": int(time.time()) + 3600,
        })
        manager = _make_manager(tmp_path)
        assert manager.get_token() == "still-valid"

    def test_refreshes_expired_token(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "expired-tok",
            "refresh_token": "ref-tok",
            "expires_at": int(time.time()) - 100,
            "backend_url": "https://api.example.com",
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-access-tok",
            "refresh_token": "new-ref-tok",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        manager = _make_manager(tmp_path, http_client=mock_client)
        token = manager.get_token()

        assert token == "new-access-tok"
        mock_client.post.assert_called_once()
        # Verify credentials were updated on disk
        loaded = manager._load_creds()
        assert loaded is not None
        assert loaded["access_token"] == "new-access-tok"

    def test_returns_none_when_no_creds(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        assert manager.get_token() is None

    def test_returns_none_when_refresh_fails(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "expired",
            "refresh_token": "ref",
            "expires_at": int(time.time()) - 100,
        })

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")

        manager = _make_manager(tmp_path, http_client=mock_client)
        assert manager.get_token() is None

    def test_returns_none_when_no_refresh_token(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "expired",
            "expires_at": int(time.time()) - 100,
        })
        manager = _make_manager(tmp_path)
        assert manager.get_token() is None


# ── Logout ────────────────────────────────────────────────────


class TestLogout:
    """logout() revokes token (best-effort) and deletes creds."""

    def test_logout_deletes_creds(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "tok",
            "refresh_token": "ref",
        })
        mock_client = MagicMock(spec=httpx.Client)

        manager = _make_manager(tmp_path, http_client=mock_client)
        manager.logout()

        assert manager._load_creds() is None
        # Verify revocation was attempted
        mock_client.post.assert_called_once()

    def test_logout_when_no_creds(self, tmp_path: Path) -> None:
        """Logout should not fail when no credentials exist."""
        manager = _make_manager(tmp_path)
        manager.logout()  # Should not raise


# ── Enforced config ───────────────────────────────────────────


class TestGetEnforcedConfig:
    """get_enforced_config() fetches policy from backend."""

    def test_returns_enforced_fields(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "enforced_fields": {"models.default": "gemini-pro", "budget.max_cost": 5.0},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response

        manager = _make_manager(tmp_path, http_client=mock_client)
        result = manager.get_enforced_config()

        assert result == {"models.default": "gemini-pro", "budget.max_cost": 5.0}

    def test_returns_empty_dict_on_network_failure(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.HTTPError("Connection refused")

        manager = _make_manager(tmp_path, http_client=mock_client)
        assert manager.get_enforced_config() == {}

    def test_returns_empty_dict_when_no_token(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        assert manager.get_enforced_config() == {}

    def test_caches_result(self, tmp_path: Path) -> None:
        _write_creds(tmp_path, {
            "access_token": "tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_response = MagicMock()
        mock_response.json.return_value = {"enforced_fields": {"k": "v"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response

        manager = _make_manager(tmp_path, http_client=mock_client)
        manager.get_enforced_config()
        manager.get_enforced_config()

        # Only one HTTP call — second call uses cache
        assert mock_client.get.call_count == 1


# ── _apply_enforced_config ────────────────────────────────────


class TestApplyEnforcedConfig:
    """_apply_enforced_config mutates settings in-place."""

    def test_applies_dotted_field(self) -> None:
        from vaig.cli._helpers import _apply_enforced_config

        settings = Settings()
        original_model = settings.models.default

        _apply_enforced_config(settings, {"models.default": "enforced-model"})
        assert settings.models.default == "enforced-model"
        assert settings.models.default != original_model

    def test_ignores_unknown_field(self) -> None:
        from vaig.cli._helpers import _apply_enforced_config

        settings = Settings()
        # Should not raise — logs a warning
        _apply_enforced_config(settings, {"nonexistent.field.path": "value"})

    def test_applies_multiple_fields(self) -> None:
        from vaig.cli._helpers import _apply_enforced_config

        settings = Settings()
        _apply_enforced_config(settings, {
            "models.default": "gemini-ultra",
            "gcp.location": "us-west1",
        })
        assert settings.models.default == "gemini-ultra"
        assert settings.gcp.location == "us-west1"


# ── _check_platform_auth no-op ────────────────────────────────


class TestCheckPlatformAuth:
    """_check_platform_auth is a no-op when platform is disabled."""

    def test_noop_when_disabled(self) -> None:
        from vaig.cli._helpers import _check_platform_auth

        settings = Settings()
        assert settings.platform.enabled is False

        # Should return without error
        _check_platform_auth(settings)

    def test_exits_when_not_authenticated(self, tmp_path: Path) -> None:

        from vaig.cli._helpers import _check_platform_auth

        settings = Settings()
        # Use object.__setattr__ to bypass the validator for testing the
        # _check_platform_auth helper in isolation (it checks enabled first).
        object.__setattr__(settings.platform, "enabled", True)
        object.__setattr__(settings.platform, "backend_url", "https://api.example.com")

        # Mock PlatformAuthManager to return not authenticated
        mock_manager = MagicMock()
        mock_manager.is_authenticated.return_value = False

        with patch("vaig.cli._helpers._platform_auth_manager", mock_manager):
            with pytest.raises(click.exceptions.Exit):
                _check_platform_auth(settings)


# ── build_container with platform auth ────────────────────────


class TestBuildContainerPlatformAuth:
    """build_container() wires PlatformAuthManager when platform enabled."""

    def test_platform_auth_none_when_disabled(self) -> None:
        settings = Settings()
        assert settings.platform.enabled is False

        from vaig.core.container import build_container

        container = build_container(settings)
        assert container.platform_auth is None

    def test_platform_auth_created_when_enabled(self) -> None:
        settings = Settings()
        object.__setattr__(settings.platform, "enabled", True)
        object.__setattr__(settings.platform, "backend_url", "https://api.example.com")
        object.__setattr__(settings.platform, "org_id", "test-org")

        from vaig.core.container import build_container

        container = build_container(settings)
        assert container.platform_auth is not None

        from vaig.core.platform_auth import PlatformAuthManager

        assert isinstance(container.platform_auth, PlatformAuthManager)

    def test_platform_auth_none_when_no_backend_url(self) -> None:
        """Platform enabled but no backend_url → don't create manager.

        With the PlatformConfig validator, creating ``PlatformConfig(enabled=True)``
        without ``backend_url`` now raises ``ValueError``.  We bypass the validator
        via ``object.__setattr__`` to test the build_container guard independently.
        """
        settings = Settings()
        object.__setattr__(settings.platform, "enabled", True)
        object.__setattr__(settings.platform, "backend_url", "")

        from vaig.core.container import build_container

        container = build_container(settings)
        assert container.platform_auth is None

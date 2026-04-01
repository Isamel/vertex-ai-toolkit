"""Integration tests for config enforcement (Phase 4, Task 4.2).

Covers:
  - SC-AUTH-004c: _apply_enforced_config mutates settings with known values
  - SC-AUTH-004d: Graceful degradation when backend unreachable (httpx.ConnectError)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx

from vaig.cli._helpers import _apply_enforced_config
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
    import json

    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps(creds), encoding="utf-8")


# ── Tests: _apply_enforced_config ─────────────────────────────


class TestApplyEnforcedConfigIntegration:
    """SC-AUTH-004c: Apply enforced config dict, assert overridden fields."""

    def test_applies_single_field(self) -> None:
        settings = Settings()
        original = settings.models.default
        _apply_enforced_config(settings, {"models.default": "gemini-2.0-flash"})
        assert settings.models.default == "gemini-2.0-flash"
        assert settings.models.default != original

    def test_applies_multiple_fields(self) -> None:
        settings = Settings()
        _apply_enforced_config(settings, {
            "models.default": "gemini-ultra",
            "gcp.location": "europe-west4",
        })
        assert settings.models.default == "gemini-ultra"
        assert settings.gcp.location == "europe-west4"

    def test_unknown_field_silently_ignored(self) -> None:
        settings = Settings()
        original_default = settings.models.default
        _apply_enforced_config(settings, {"nonexistent.deeply.nested.field": "value"})
        # Should not raise and should not change anything
        assert settings.models.default == original_default

    def test_partial_path_failure_does_not_affect_valid(self) -> None:
        """Mix of valid and invalid paths — valid ones apply."""
        settings = Settings()
        _apply_enforced_config(settings, {
            "models.default": "enforced-model",
            "bogus.path": "ignored",
        })
        assert settings.models.default == "enforced-model"

    def test_enforced_overrides_user_set_value(self) -> None:
        """Enforced config overrides even explicitly-set user values."""
        settings = Settings()
        settings.models.default = "user-choice"
        _apply_enforced_config(settings, {"models.default": "admin-enforced"})
        assert settings.models.default == "admin-enforced"


# ── Tests: Graceful degradation on backend failure ────────────


class TestGracefulDegradation:
    """SC-AUTH-004d: Graceful degradation when backend unreachable."""

    def test_connect_error_returns_empty_dict(self, tmp_path: Path) -> None:
        """Backend unreachable → get_enforced_config() returns {}."""
        _write_creds(tmp_path, {
            "access_token": "valid-tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        manager = _make_manager(tmp_path, http_client=mock_client)
        result = manager.get_enforced_config()

        assert result == {}

    def test_timeout_error_returns_empty_dict(self, tmp_path: Path) -> None:
        """Backend timeout → get_enforced_config() returns {}."""
        _write_creds(tmp_path, {
            "access_token": "valid-tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ReadTimeout("Timed out")

        manager = _make_manager(tmp_path, http_client=mock_client)
        result = manager.get_enforced_config()

        assert result == {}

    def test_http_500_returns_empty_dict(self, tmp_path: Path) -> None:
        """Backend 500 → get_enforced_config() returns {}."""
        _write_creds(tmp_path, {
            "access_token": "valid-tok",
            "expires_at": int(time.time()) + 3600,
        })

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response

        manager = _make_manager(tmp_path, http_client=mock_client)
        result = manager.get_enforced_config()

        assert result == {}

    def test_no_token_returns_empty_dict(self, tmp_path: Path) -> None:
        """No credentials → get_enforced_config() returns {} without HTTP call."""
        mock_client = MagicMock(spec=httpx.Client)
        manager = _make_manager(tmp_path, http_client=mock_client)

        result = manager.get_enforced_config()

        assert result == {}
        mock_client.get.assert_not_called()

"""Integration tests for CLI auth commands (Phase 4, Task 4.3).

Covers:
  - SC-AUTH-003a: whoami output format (Email, Org, Role, CLI ID)
  - SC-AUTH-003b: status output format (authenticated + config policy)
  - SC-AUTH-003c: Auth commands exit code 1 when platform disabled
  - SC-AUTH-003d: logout when not authenticated (prints message, no error)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from vaig.cli.commands import auth as auth_module

runner = CliRunner()


# ── Helpers ────────────────────────────────────────────────────


def _make_test_app() -> typer.Typer:
    """Create a fresh Typer app with auth commands registered."""
    app = typer.Typer()
    auth_module.register(app)
    return app


def _mock_settings(*, platform_enabled: bool = True) -> MagicMock:
    """Create a mock Settings object with platform config."""
    settings = MagicMock()
    settings.platform.enabled = platform_enabled
    settings.platform.backend_url = "https://api.example.com"
    settings.platform.org_id = "test-org"
    return settings


def _mock_auth_manager(
    *,
    is_authenticated: bool = True,
    user_info: dict[str, Any] | None = None,
    enforced_config: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock PlatformAuthManager."""
    manager = MagicMock()
    manager.is_authenticated.return_value = is_authenticated
    manager.get_user_info.return_value = user_info or {
        "email": "alice@example.com",
        "org_id": "test-org",
        "role": "admin",
        "cli_id": "cli-abc123",
    }
    manager.get_enforced_config.return_value = enforced_config or {}
    return manager


# ── Tests: Platform disabled → exit code 1 ───────────────────


class TestPlatformDisabled:
    """SC-AUTH-003c: All auth commands exit 1 when platform not enabled."""

    def test_login_exits_when_platform_disabled(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=False)
        with patch.object(auth_module, "_get_settings", return_value=settings):
            result = runner.invoke(app, ["login"])
        assert result.exit_code != 0

    def test_logout_exits_when_platform_disabled(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=False)
        with patch.object(auth_module, "_get_settings", return_value=settings):
            result = runner.invoke(app, ["logout"])
        assert result.exit_code != 0

    def test_whoami_exits_when_platform_disabled(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=False)
        with patch.object(auth_module, "_get_settings", return_value=settings):
            result = runner.invoke(app, ["whoami"])
        assert result.exit_code != 0

    def test_status_exits_when_platform_disabled(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=False)
        with patch.object(auth_module, "_get_settings", return_value=settings):
            result = runner.invoke(app, ["status"])
        assert result.exit_code != 0


# ── Tests: whoami output format ──────────────────────────────


class TestWhoami:
    """SC-AUTH-003a: whoami shows Email, Org, Role, CLI ID."""

    def test_whoami_output_format(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(
            is_authenticated=True,
            user_info={
                "email": "alice@example.com",
                "org_id": "my-org",
                "role": "admin",
                "cli_id": "cli-xyz",
            },
        )

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["whoami"])

        assert result.exit_code == 0
        assert "alice@example.com" in result.output
        assert "my-org" in result.output
        assert "admin" in result.output
        assert "cli-xyz" in result.output

    def test_whoami_not_authenticated(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(is_authenticated=False)

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["whoami"])

        assert result.exit_code != 0


# ── Tests: status output format ──────────────────────────────


class TestStatus:
    """SC-AUTH-003b: status shows authenticated, email, org, role, CLI ID, config policy."""

    def test_status_output_with_enforced_config(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(
            is_authenticated=True,
            user_info={
                "email": "bob@corp.com",
                "org_id": "corp-org",
                "role": "operator",
                "cli_id": "cli-op1",
            },
            enforced_config={"models.default": "gemini-pro"},
        )

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "authenticated" in result.output
        assert "bob@corp.com" in result.output
        assert "corp-org" in result.output
        assert "operator" in result.output
        assert "cli-op1" in result.output
        assert "active" in result.output

    def test_status_output_without_enforced_config(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(
            is_authenticated=True,
            enforced_config={},
        )

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "no enforced fields" in result.output

    def test_status_not_authenticated(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(is_authenticated=False)

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code != 0


# ── Tests: logout when not authenticated ─────────────────────


class TestLogout:
    """SC-AUTH-003d: logout when not authenticated prints message, no error."""

    def test_logout_not_authenticated_succeeds(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(is_authenticated=False)

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["logout"])

        # logout prints "Not currently authenticated" and returns normally (no exit code 1)
        assert result.exit_code == 0
        assert "Not currently authenticated" in result.output

    def test_logout_authenticated_succeeds(self) -> None:
        app = _make_test_app()
        settings = _mock_settings(platform_enabled=True)
        manager = _mock_auth_manager(is_authenticated=True)

        with (
            patch.object(auth_module, "_get_settings", return_value=settings),
            patch.object(auth_module, "_get_auth_manager", return_value=manager),
        ):
            result = runner.invoke(app, ["logout"])

        assert result.exit_code == 0
        assert "Logged out successfully" in result.output
        manager.logout.assert_called_once()

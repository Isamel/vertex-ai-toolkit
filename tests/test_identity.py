"""Tests for the identity resolution module (vaig.core.identity).

Covers:
- OS user resolution with getpass mock
- GCP user resolution for service accounts, user credentials, and fallbacks
- Composite key construction
- Session-level caching of resolved identity
- App version retrieval
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.identity import (
    UNKNOWN_GCP_USER,
    UNKNOWN_OS_USER,
    _reset_identity_cache,
    build_composite_key,
    get_app_version,
    resolve_gcp_user,
    resolve_identity,
    resolve_os_user,
)


@pytest.fixture(autouse=True)
def _clean_identity_cache() -> None:
    """Reset the identity cache before each test to avoid cross-test leakage."""
    _reset_identity_cache()


# ── OS user tests ────────────────────────────────────────────


class TestResolveOsUser:
    """Tests for resolve_os_user()."""

    @patch("vaig.core.identity.getpass.getuser", return_value="testuser")
    def test_returns_string(self, mock_getuser: MagicMock) -> None:
        """Returns the OS username from getpass."""
        result = resolve_os_user()
        assert result == "testuser"
        mock_getuser.assert_called_once()

    @patch("vaig.core.identity.getpass.getuser", side_effect=OSError("no tty"))
    def test_fallback_on_error(self, mock_getuser: MagicMock) -> None:
        """Falls back to UNKNOWN_OS_USER when getpass raises."""
        result = resolve_os_user()
        assert result == UNKNOWN_OS_USER


# ── GCP user tests ───────────────────────────────────────────


class TestResolveGcpUser:
    """Tests for resolve_gcp_user()."""

    def test_service_account(self) -> None:
        """Resolves email from service_account_email attribute."""
        creds = MagicMock()
        creds.service_account_email = "sa@project.iam.gserviceaccount.com"
        result = resolve_gcp_user(creds)
        assert result == "sa@project.iam.gserviceaccount.com"

    def test_user_credentials_account(self) -> None:
        """Resolves email from account attribute (user credentials)."""
        creds = MagicMock(spec=[])  # empty spec so getattr falls through
        creds.account = "user@example.com"
        # Ensure service_account_email is not present
        result = resolve_gcp_user(creds)
        assert result == "user@example.com"

    def test_none_credentials(self) -> None:
        """Returns UNKNOWN_GCP_USER when credentials is None."""
        result = resolve_gcp_user(None)
        assert result == UNKNOWN_GCP_USER

    def test_fallback_no_useful_attrs(self) -> None:
        """Returns UNKNOWN_GCP_USER when credentials have no email-like attrs."""
        creds = MagicMock(spec=[])  # empty spec — no attributes
        result = resolve_gcp_user(creds)
        assert result == UNKNOWN_GCP_USER

    def test_service_account_email_default_ignored(self) -> None:
        """Skips service_account_email when it's the literal 'default'."""
        creds = MagicMock(spec=[])
        creds.service_account_email = "default"
        result = resolve_gcp_user(creds)
        assert result == UNKNOWN_GCP_USER

    def test_signer_email_fallback(self) -> None:
        """Resolves email from signer_email as last resort."""
        creds = MagicMock(spec=[])
        creds.signer_email = "signer@project.iam.gserviceaccount.com"
        result = resolve_gcp_user(creds)
        assert result == "signer@project.iam.gserviceaccount.com"


# ── Composite key tests ──────────────────────────────────────


class TestBuildCompositeKey:
    """Tests for build_composite_key()."""

    def test_format(self) -> None:
        """Composite key follows '{os_user}:{gcp_user}' format."""
        key = build_composite_key("alice", "alice@example.com")
        assert key == "alice:alice@example.com"

    def test_unknown_fallbacks(self) -> None:
        """Works correctly with unknown fallback values."""
        key = build_composite_key(UNKNOWN_OS_USER, UNKNOWN_GCP_USER)
        assert key == f"{UNKNOWN_OS_USER}:{UNKNOWN_GCP_USER}"


# ── Caching tests ────────────────────────────────────────────


class TestResolveIdentity:
    """Tests for resolve_identity() caching behaviour."""

    @patch("vaig.core.identity.resolve_gcp_user", return_value="user@example.com")
    @patch("vaig.core.identity.resolve_os_user", return_value="testuser")
    def test_caching(
        self, mock_os: MagicMock, mock_gcp: MagicMock
    ) -> None:
        """Calling resolve_identity twice only resolves once (cached)."""
        creds = MagicMock()

        result1 = resolve_identity(creds)
        result2 = resolve_identity(creds)

        assert result1 == result2
        assert result1 == ("testuser", "user@example.com", "testuser:user@example.com")
        # Each resolver called exactly once despite two calls
        mock_os.assert_called_once()
        mock_gcp.assert_called_once()

    @patch("vaig.core.identity.resolve_gcp_user", return_value=UNKNOWN_GCP_USER)
    @patch("vaig.core.identity.resolve_os_user", return_value="bob")
    def test_returns_tuple(self, mock_os: MagicMock, mock_gcp: MagicMock) -> None:
        """Returns a 3-tuple of (os_user, gcp_user, composite_key)."""
        os_user, gcp_user, key = resolve_identity(None)
        assert os_user == "bob"
        assert gcp_user == UNKNOWN_GCP_USER
        assert key == f"bob:{UNKNOWN_GCP_USER}"


# ── App version tests ────────────────────────────────────────


class TestGetAppVersion:
    """Tests for get_app_version()."""

    def test_returns_version_string(self) -> None:
        """Returns a non-empty version string."""
        version = get_app_version()
        assert isinstance(version, str)
        assert len(version) > 0

    @patch("vaig.__version__", "1.2.3")
    def test_returns_known_version(self) -> None:
        """Returns the expected version when __version__ is set."""
        version = get_app_version()
        assert version == "1.2.3"

    @patch.dict("sys.modules", {"vaig": None})
    def test_fallback_on_import_error(self) -> None:
        """Returns 'unknown' when vaig cannot be imported."""
        # Force ImportError by removing vaig from sys.modules
        version = get_app_version()
        # When vaig is None in sys.modules, importing from it raises ImportError
        assert version == "unknown"

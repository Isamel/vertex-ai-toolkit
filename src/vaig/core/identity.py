"""Identity resolution for audit logging and rate limiting.

Resolves the current user's OS username and GCP authenticated identity
for use in audit records and quota enforcement.
"""

from __future__ import annotations

import getpass
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

logger = logging.getLogger(__name__)

UNKNOWN_OS_USER = "unknown_os_user"
UNKNOWN_GCP_USER = "unknown_gcp_user"

# Module-level cache — resolved once per process (identity doesn't change mid-session).
_cached_identity: tuple[str, str, str] | None = None


def resolve_os_user() -> str:
    """Return the current OS username, or fallback constant."""
    try:
        return getpass.getuser()
    except Exception:
        logger.warning("Could not resolve OS user, using fallback")
        return UNKNOWN_OS_USER


def resolve_gcp_user(credentials: Credentials | None = None) -> str:
    """Return the GCP authenticated user email from credentials.

    Resolution order:
    1. Service account: credentials.service_account_email
    2. Common credential attributes with email-like values
    3. Fallback: UNKNOWN_GCP_USER
    """
    if credentials is None:
        return UNKNOWN_GCP_USER

    # Service account credentials have .service_account_email
    email = getattr(credentials, "service_account_email", None)
    if email and email != "default":
        return str(email)

    # User credentials — try common attributes that hold the account email.
    # Lazy imports avoid pulling heavy GCP packages at module level.
    try:
        account = getattr(credentials, "account", None) or getattr(
            credentials, "_account", None
        )
        if account and isinstance(account, str) and "@" in account:
            return str(account)
    except Exception:
        logger.debug("Could not resolve GCP user from account attribute")

    # Last resort: check for signer-related email attributes
    for attr in ("signer_email", "_signer_email"):
        val = getattr(credentials, attr, None)
        if val and isinstance(val, str) and "@" in val:
            return str(val)

    return UNKNOWN_GCP_USER


def build_composite_key(os_user: str, gcp_user: str) -> str:
    """Build composite user key for rate limiting: '{os_user}:{gcp_email}'."""
    return f"{os_user}:{gcp_user}"


def resolve_identity(
    credentials: Credentials | None = None,
) -> tuple[str, str, str]:
    """Resolve full identity once and cache for the session.

    Returns:
        Tuple of (os_user, gcp_user, composite_key)
    """
    global _cached_identity  # noqa: PLW0603
    if _cached_identity is not None:
        return _cached_identity

    os_user = resolve_os_user()
    gcp_user = resolve_gcp_user(credentials)
    result = (os_user, gcp_user, build_composite_key(os_user, gcp_user))
    # Only cache if GCP user was actually resolved — future calls with
    # credentials should get a chance to resolve properly.
    if gcp_user != UNKNOWN_GCP_USER:
        _cached_identity = result
    return result


def _reset_identity_cache() -> None:
    """Reset cached identity — for testing only."""
    global _cached_identity  # noqa: PLW0603
    _cached_identity = None


def get_app_version() -> str:
    """Return the current vaig app version."""
    try:
        from vaig import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"

"""Authentication module — ADC, gcloud CLI fallback, and SA impersonation."""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials
from google.oauth2.credentials import Credentials as OAuth2Credentials

from vaig.core.exceptions import GCPAuthError, GCPPermissionError

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Scopes required for Vertex AI
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# gcloud access tokens typically expire after 1 hour
_GCLOUD_TOKEN_LIFETIME = timedelta(hours=1)


def get_credentials(settings: Settings) -> Credentials:
    """Get GCP credentials based on the configured auth mode.

    - ADC: Uses Application Default Credentials (Pod SA or gcloud auth).
    - Impersonate: Uses local credentials to impersonate a Service Account.

    Falls back to gcloud CLI access token when ADC is not configured.
    """
    if settings.auth.mode == "impersonate":
        return _get_impersonated_credentials(settings.auth.impersonate_sa)
    return _get_adc_credentials()


def get_gke_credentials(settings: Settings) -> Credentials | None:
    """Get GCP credentials for GKE/observability APIs (Cloud Logging, Monitoring, GKE API).

    Implements a 3-tier fallback:
    1. If ``gke.impersonate_sa`` is set → impersonate that SA (GKE-specific override).
    2. Elif ``auth.mode == "impersonate"`` and ``auth.impersonate_sa`` is set → reuse that SA.
    3. Else → return ``None`` (let GCP clients use ADC default).

    Returning ``None`` preserves backward compatibility: GCP client constructors
    treat ``credentials=None`` as "use Application Default Credentials".
    """
    gke_sa = settings.gke.impersonate_sa
    if gke_sa:
        logger.info("GKE credentials: impersonating GKE-specific SA %s", gke_sa)
        return _get_impersonated_credentials(gke_sa)

    if settings.auth.mode == "impersonate" and settings.auth.impersonate_sa:
        logger.info(
            "GKE credentials: reusing auth.impersonate_sa %s",
            settings.auth.impersonate_sa,
        )
        return _get_impersonated_credentials(settings.auth.impersonate_sa)

    logger.info("GKE credentials: using ADC (no SA impersonation configured)")
    return None


def _get_adc_credentials() -> Credentials:
    """Get Application Default Credentials with gcloud CLI fallback.

    Priority:
    1. ADC (GKE Workload Identity, ``gcloud auth application-default login``)
    2. gcloud CLI access token (``gcloud auth print-access-token``)

    The fallback is useful in development environments (Codespaces,
    devcontainers) where ADC is not configured but ``gcloud auth login``
    has been run.

    Raises:
        GCPAuthError: If neither ADC nor gcloud CLI credentials can be obtained.
    """
    try:
        credentials, project = google.auth.default(scopes=_VERTEX_SCOPES)
        logger.info("Using ADC credentials (project: %s)", project)
        return credentials
    except google.auth.exceptions.DefaultCredentialsError:
        logger.info("ADC not found, falling back to gcloud CLI access token")
        try:
            return _get_gcloud_token_credentials()
        except RuntimeError as exc:
            raise GCPAuthError(
                f"No GCP credentials available: {exc}",
                fix_suggestion=(
                    "Run one of:\n"
                    "  1. gcloud auth application-default login\n"
                    "  2. gcloud auth login\n"
                    "  3. Set GOOGLE_APPLICATION_CREDENTIALS env var"
                ),
            ) from exc


def _fetch_gcloud_access_token() -> str:
    """Run ``gcloud auth print-access-token`` and return the raw token string.

    Raises:
        RuntimeError: If gcloud is not installed, not authenticated, or
            the command fails for any reason.
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        msg = "gcloud CLI not found. Install Google Cloud SDK or set up ADC."
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = "gcloud auth print-access-token timed out after 10s."
        raise RuntimeError(msg) from exc

    token = (result.stdout or "").strip()

    if not token or result.returncode != 0:
        stderr_hint = result.stderr.strip() if result.stderr else ""
        msg = (
            "Could not obtain credentials. Either:\n"
            "  1. Run: gcloud auth application-default login\n"
            "  2. Run: gcloud auth login\n"
            "  3. Set GOOGLE_APPLICATION_CREDENTIALS env var"
        )
        if stderr_hint:
            msg += f"\n\ngcloud stderr: {stderr_hint}"
        raise RuntimeError(msg)

    return token


def _gcloud_refresh_handler(
    request: Any,  # noqa: ARG001 — unused but required by google-auth API
    scopes: Any = None,  # noqa: ARG001
) -> tuple[str, datetime]:
    """Refresh handler that fetches a fresh token via gcloud CLI.

    This callback is invoked by ``google.oauth2.credentials.Credentials.refresh()``
    when the current access token is expired or about to expire.  It is
    thread-safe: ``google-auth`` serialises calls to ``before_request()``
    internally, and our ``_fetch_gcloud_access_token()`` is stateless.

    Returns:
        Tuple of (access_token, expiry_datetime_utc).

    Raises:
        google.auth.exceptions.RefreshError: If the token cannot be refreshed
            (wraps the underlying RuntimeError with a user-friendly message).
    """
    try:
        token = _fetch_gcloud_access_token()
    except RuntimeError as exc:
        raise auth_exceptions.RefreshError(  # type: ignore[no-untyped-call]
            "Failed to refresh gcloud access token. "
            "Please re-authenticate with: gcloud auth login"
        ) from exc

    # google-auth internally uses naive UTC datetimes (via _helpers.utcnow()),
    # so the refresh_handler must return a naive UTC expiry to avoid
    # comparison errors in Credentials.refresh().
    expiry = datetime.now(tz=UTC).replace(tzinfo=None) + _GCLOUD_TOKEN_LIFETIME
    logger.debug("gcloud token refreshed, new expiry: %s", expiry)
    return token, expiry


def _get_gcloud_token_credentials() -> Credentials:
    """Get credentials from the active gcloud CLI session with auto-refresh.

    Runs ``gcloud auth print-access-token`` to obtain a short-lived
    OAuth2 access token.  Unlike the previous implementation, this version
    registers a ``refresh_handler`` so that the google-auth library
    automatically re-fetches a fresh token when the current one expires
    (typically after 1 hour).

    The refresh is handled transparently by ``Credentials.before_request()``,
    which checks token expiry before every API call.  The ``refresh_handler``
    callback runs ``gcloud auth print-access-token`` again to obtain a new
    token — no user interaction required as long as the gcloud session is
    still valid.
    """
    token = _fetch_gcloud_access_token()
    # Use naive UTC to match google-auth's internal convention
    expiry = datetime.now(tz=UTC).replace(tzinfo=None) + _GCLOUD_TOKEN_LIFETIME

    credentials = OAuth2Credentials(  # type: ignore[no-untyped-call]
        token=token,
        expiry=expiry,
        refresh_handler=_gcloud_refresh_handler,
    )

    logger.info("Using gcloud CLI access token (auto-refresh enabled, ~1h lifetime)")
    return credentials


def _get_impersonated_credentials(
    target_sa: str,
    scopes: list[str] | None = None,
) -> Credentials:
    """Impersonate a Service Account using local credentials.

    This is the recommended approach for local development:
    1. Developer has their own GCP credentials (via gcloud auth)
    2. Those credentials impersonate the target Service Account
    3. All API calls are made AS the Service Account

    The developer needs `roles/iam.serviceAccountTokenCreator` on the target SA.

    Args:
        target_sa: Service account email to impersonate.
        scopes: OAuth2 scopes for the impersonated credential.
            Defaults to ``_VERTEX_SCOPES`` (``cloud-platform``).

    Raises:
        GCPAuthError: If source credentials cannot be obtained.
        GCPPermissionError: If impersonation fails due to missing permissions.
    """
    if not target_sa:
        msg = (
            "SA impersonation requires VAIG_IMPERSONATE_SA to be set. "
            "Example: my-sa@my-project.iam.gserviceaccount.com"
        )
        raise ValueError(msg)

    # Get source credentials (developer's own)
    try:
        source_credentials, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise GCPAuthError(
            "Cannot obtain source credentials for SA impersonation.",
            fix_suggestion="Run: gcloud auth application-default login",
        ) from exc

    try:
        # Create impersonated credentials
        target_credentials = impersonated_credentials.Credentials(  # type: ignore[no-untyped-call]
            source_credentials=source_credentials,
            target_principal=target_sa,
            target_scopes=scopes or _VERTEX_SCOPES,
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "permission" in exc_str or "403" in exc_str or "iam" in exc_str:
            raise GCPPermissionError(
                f"Cannot impersonate SA '{target_sa}': {exc}",
                required_permissions=["roles/iam.serviceAccountTokenCreator"],
                fix_suggestion=(
                    f"Grant roles/iam.serviceAccountTokenCreator on {target_sa} "
                    "to your user account"
                ),
            ) from exc
        raise

    logger.info("Impersonating SA: %s", target_sa)
    return target_credentials

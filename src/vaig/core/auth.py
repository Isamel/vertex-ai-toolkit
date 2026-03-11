"""Authentication module — ADC, gcloud CLI fallback, and SA impersonation."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import google.auth
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials
from google.oauth2.credentials import Credentials as OAuth2Credentials

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Scopes required for Vertex AI
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_credentials(settings: Settings) -> Credentials:
    """Get GCP credentials based on the configured auth mode.

    - ADC: Uses Application Default Credentials (Pod SA or gcloud auth).
    - Impersonate: Uses local credentials to impersonate a Service Account.

    Falls back to gcloud CLI access token when ADC is not configured.
    """
    if settings.auth.mode == "impersonate":
        return _get_impersonated_credentials(settings.auth.impersonate_sa)
    return _get_adc_credentials()


def _get_adc_credentials() -> Credentials:
    """Get Application Default Credentials with gcloud CLI fallback.

    Priority:
    1. ADC (GKE Workload Identity, ``gcloud auth application-default login``)
    2. gcloud CLI access token (``gcloud auth print-access-token``)

    The fallback is useful in development environments (Codespaces,
    devcontainers) where ADC is not configured but ``gcloud auth login``
    has been run.
    """
    try:
        credentials, project = google.auth.default(scopes=_VERTEX_SCOPES)
        logger.info("Using ADC credentials (project: %s)", project)
        return credentials
    except google.auth.exceptions.DefaultCredentialsError:
        logger.info("ADC not found, falling back to gcloud CLI access token")
        return _get_gcloud_token_credentials()


def _get_gcloud_token_credentials() -> Credentials:
    """Get credentials from the active gcloud CLI session.

    Runs ``gcloud auth print-access-token`` to obtain a short-lived
    OAuth2 access token.  This is a convenience fallback — the token
    is NOT automatically refreshed and will expire (typically 1 h).
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        token = result.stdout.strip()

        if not token or result.returncode != 0:
            msg = (
                "Could not obtain credentials. Either:\n"
                "  1. Run: gcloud auth application-default login\n"
                "  2. Run: gcloud auth login\n"
                "  3. Set GOOGLE_APPLICATION_CREDENTIALS env var"
            )
            raise RuntimeError(msg)

        logger.info("Using gcloud CLI access token (short-lived, ~1h)")
        return OAuth2Credentials(token=token)

    except FileNotFoundError as exc:
        msg = "gcloud CLI not found. Install Google Cloud SDK or set up ADC."
        raise RuntimeError(msg) from exc


def _get_impersonated_credentials(target_sa: str) -> Credentials:
    """Impersonate a Service Account using local credentials.

    This is the recommended approach for local development:
    1. Developer has their own GCP credentials (via gcloud auth)
    2. Those credentials impersonate the target Service Account
    3. All API calls are made AS the Service Account

    The developer needs `roles/iam.serviceAccountTokenCreator` on the target SA.
    """
    if not target_sa:
        msg = (
            "SA impersonation requires VAIG_IMPERSONATE_SA to be set. "
            "Example: my-sa@my-project.iam.gserviceaccount.com"
        )
        raise ValueError(msg)

    # Get source credentials (developer's own)
    source_credentials, _ = google.auth.default()

    # Create impersonated credentials
    target_credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target_sa,
        target_scopes=_VERTEX_SCOPES,
    )

    logger.info("Impersonating SA: %s", target_sa)
    return target_credentials

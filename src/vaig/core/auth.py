"""Authentication module — ADC and Service Account impersonation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import google.auth
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Scopes required for Vertex AI
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_credentials(settings: Settings) -> Credentials:
    """Get GCP credentials based on the configured auth mode.

    - ADC: Uses Application Default Credentials (Pod SA or gcloud auth).
    - Impersonate: Uses local credentials to impersonate a Service Account.
    """
    if settings.auth.mode == "impersonate":
        return _get_impersonated_credentials(settings.auth.impersonate_sa)
    return _get_adc_credentials()


def _get_adc_credentials() -> Credentials:
    """Get Application Default Credentials.

    In a GKE Pod with Workload Identity, this automatically picks up
    the Kubernetes Service Account → GCP Service Account mapping.

    Locally, uses whatever `gcloud auth application-default login` set up.
    """
    credentials, project = google.auth.default(scopes=_VERTEX_SCOPES)
    logger.info("Using ADC credentials (project: %s)", project)
    return credentials


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

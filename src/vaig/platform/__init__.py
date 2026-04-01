"""Platform authentication, management, and admin portal package.

This package contains the platform backend (FastAPI), Pydantic models
for organizations/users/CLI instances, and supporting services (JWT,
Firestore).  All platform functionality is gated behind
``settings.platform.enabled`` — when disabled, this package is never
imported by the CLI.
"""

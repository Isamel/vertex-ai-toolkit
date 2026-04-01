"""FastAPI application factory for the platform admin backend.

This is a SEPARATE application from the web UI (``vaig.web.app``).
The platform app serves JSON APIs for authentication, CLI management,
and configuration policy enforcement.

Usage::

    from vaig.platform.app import create_platform_app
    app = create_platform_app()
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vaig import __version__
from vaig.platform.api.auth import router as auth_router
from vaig.platform.api.cli import router as cli_router
from vaig.platform.api.config_policy import router as config_router
from vaig.platform.core.firestore import AbstractRepository, InMemoryRepository
from vaig.platform.core.jwt import JWTService

logger = logging.getLogger(__name__)


def create_platform_app(
    *,
    jwt_service: JWTService | None = None,
    repository: AbstractRepository | None = None,
) -> FastAPI:
    """Build and configure the platform FastAPI application.

    Mounts API routers under ``/api/v1/`` and a ``GET /healthz``
    health check at the root.

    Args:
        jwt_service: Optional pre-configured JWTService. If ``None``,
            a default instance is created (reads keys from env vars).
        repository: Optional repository for data persistence. If ``None``,
            an in-memory repository is used (suitable for development).

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="VAIG Platform API",
        description="Vertex AI Gemini Toolkit — Platform Admin Backend",
        version=__version__,
    )

    # Store services on app.state for dependency injection
    app.state.jwt_service = jwt_service or JWTService()
    app.state.repository = repository or InMemoryRepository()

    # ── Health check ──────────────────────────────────────────

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ── API v1 routers ────────────────────────────────────────
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(cli_router, prefix="/api/v1")
    app.include_router(config_router, prefix="/api/v1")

    return app

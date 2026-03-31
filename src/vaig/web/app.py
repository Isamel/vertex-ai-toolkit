"""FastAPI application factory for the VAIG web interface."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vaig import __version__

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Mounts static files, configures Jinja2 templates, and includes
    all route routers.
    """
    app = FastAPI(
        title="VAIG Web",
        description="Vertex AI Gemini Toolkit — Web Interface",
        version=__version__,
    )

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Jinja2 templates — stored on app.state for route access
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Register routes
    from vaig.web.routes.health import router as health_router
    from vaig.web.routes.pages import router as pages_router

    app.include_router(health_router)
    app.include_router(pages_router)

    return app

"""FastAPI application factory for the VAIG web interface."""

from __future__ import annotations

import logging
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, JSONResponse, Response

from vaig import __version__

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the scheduler engine.

    If the ``schedule`` section is enabled in settings, the engine is
    started automatically and attached to ``app.state.scheduler_engine``.
    On shutdown the engine is stopped gracefully.
    """
    engine = None
    try:
        from vaig.core.config import get_settings

        settings = get_settings()
        if settings.schedule.enabled:
            from vaig.core.scheduler import SchedulerEngine

            engine = SchedulerEngine(settings)
            await engine.start()
            app.state.scheduler_engine = engine
            logger.info("Scheduler engine attached to app (lifespan)")
    except Exception:  # noqa: BLE001
        logger.warning("Scheduler engine not started — schedule feature unavailable", exc_info=True)

    yield

    if engine is not None:
        await engine.stop()
        logger.info("Scheduler engine stopped (lifespan)")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Mounts static files, configures Jinja2 templates, and includes
    all route routers.
    """
    app = FastAPI(
        title="VAIG Web",
        description="Vertex AI Gemini Toolkit — Web Interface",
        version=__version__,
        lifespan=_lifespan,
    )

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Jinja2 templates — stored on app.state for route access
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Register routes
    from vaig.web.routes.ask import router as ask_router
    from vaig.web.routes.chat import router as chat_router
    from vaig.web.routes.health import router as health_router
    from vaig.web.routes.live import router as live_router
    from vaig.web.routes.pages import router as pages_router
    from vaig.web.routes.schedules import router as schedules_router
    from vaig.web.routes.settings import router as settings_router

    app.include_router(health_router)
    app.include_router(pages_router)
    app.include_router(ask_router)
    app.include_router(chat_router)
    app.include_router(settings_router)
    app.include_router(live_router)
    app.include_router(schedules_router)

    # Ollama-compatible proxy — always registered so that Ollama
    # clients receive a JSON 404 when disabled instead of HTML.
    from vaig.web.routes.ollama import router as ollama_router

    app.include_router(ollama_router)

    # Session store — lazily initialised on first use by chat routes.
    # The store is set to ``None`` here; a concrete implementation
    # (e.g. FirestoreSessionStore) can be attached via middleware or
    # a startup event when running in production.
    app.state.session_store = None

    # Custom error handlers
    app.add_exception_handler(404, _not_found_handler)
    app.add_exception_handler(500, _server_error_handler)

    return app


async def _not_found_handler(request: Request, exc: Exception) -> Response:
    """Render a styled 404 page, or JSON for API clients."""
    # Ollama / API clients expect JSON, not HTML
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "not found"}, status_code=404)

    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="404.html",
        status_code=404,
    )


async def _server_error_handler(request: Request, exc: Exception) -> Response:
    """Render a styled 500 error page — no tracebacks unless debug mode."""
    is_debug = getattr(request.app, "debug", False)

    if is_debug:
        logger.exception("Internal server error on %s %s", request.method, request.url.path)
    else:
        logger.error(
            "Internal server error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=False,
        )

    # Only include debug info when app is in debug mode
    debug_info: str | None = None
    if is_debug:
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        debug_info = "".join(tb_lines)

    templates: Jinja2Templates = request.app.state.templates
    try:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "status_code": 500,
                "error_title": "Internal Server Error",
                "error_detail": "Something went wrong. Please try again later.",
                "debug_info": debug_info,
            },
            status_code=500,
        )
    except Exception:  # noqa: BLE001
        # Last resort — if template rendering itself fails, return plain HTML
        return HTMLResponse(
            content=(
                "<h1>500 — Internal Server Error</h1>"
                "<p>Something went wrong. Please try again later.</p>"
                '<p><a href="/">Go Home</a></p>'
            ),
            status_code=500,
        )

"""Page routes — serves HTML templates via Jinja2."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import Response

router = APIRouter(tags=["pages"])


@router.get("/")
async def index(request: Request) -> Response:
    """Render the landing page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="index.html",
    )

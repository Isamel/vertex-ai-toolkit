"""Page routes — serves HTML templates via Jinja2."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["pages"])


@router.get("/")
async def index(request: Request) -> object:
    """Render the landing page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

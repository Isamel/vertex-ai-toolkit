"""Health endpoint — liveness/readiness check."""

from __future__ import annotations

from fastapi import APIRouter

from vaig import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Return service health status and version."""
    return {"status": "ok", "version": __version__}

"""Review routes — submit and query report reviews.

``GET  /api/reviews/{run_id}``   — get review for a run (404 if none)
``POST /api/reviews/{run_id}``   — create or update review (admin-only)
``GET  /api/reviews``            — list reviews, optional ``?status=`` filter
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from vaig.core.review_store import ReviewStore
from vaig.skills.service_health.schema import (
    FindingReview,
    ReportReview,
    ReviewStatus,
)
from vaig.web.deps import get_current_user, is_admin

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reviews"])

# Module-level store instance — created once, reused across requests.
_store = ReviewStore()


# -- Request Models --------------------------------------------------------


class SubmitReviewRequest(BaseModel):
    """Body for POST /api/reviews/{run_id}."""

    status: ReviewStatus
    finding_reviews: list[FindingReview] = Field(default_factory=list)
    overall_comment: str = ""


# -- Routes ----------------------------------------------------------------


@router.get("/api/reviews/{run_id}")
async def get_review(run_id: str) -> JSONResponse:
    """Return the review for *run_id*, or 404 if none exists."""
    review = _store.get_by_run_id(run_id)
    if review is None:
        return JSONResponse(
            {"error": f"No review found for run_id {run_id!r}"},
            status_code=404,
        )
    return JSONResponse(review.model_dump(mode="json"))


@router.post("/api/reviews/{run_id}")
async def submit_review(
    run_id: str,
    body: SubmitReviewRequest,
    request: Request,
) -> JSONResponse:
    """Create or update a review for *run_id*.  Admin-only."""
    if not is_admin(request):
        return JSONResponse(
            {"error": "Admin privileges required to submit reviews"},
            status_code=403,
        )

    reviewer = get_current_user(request)
    now = datetime.now(UTC)

    existing = _store.get_by_run_id(run_id)
    if existing is not None:
        # Update — preserve submitted_at, bump updated_at.
        review = existing.model_copy(
            update={
                "status": body.status,
                "reviewer": reviewer,
                "finding_reviews": body.finding_reviews,
                "overall_comment": body.overall_comment,
                "updated_at": now,
            }
        )
    else:
        review = ReportReview(
            run_id=run_id,
            status=body.status,
            reviewer=reviewer,
            finding_reviews=body.finding_reviews,
            overall_comment=body.overall_comment,
            submitted_at=now,
            updated_at=now,
        )

    _store.save(review)

    # Fire event for audit trail.
    try:
        from vaig.core.event_bus import EventBus
        from vaig.core.events import ReportReviewed

        EventBus.get().emit(
            ReportReviewed(
                run_id=run_id,
                status=body.status.value,
                reviewer=reviewer,
            )
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.debug("Failed to emit ReportReviewed event", exc_info=True)

    return JSONResponse(review.model_dump(mode="json"), status_code=200)


@router.get("/api/reviews")
async def list_reviews(status: str | None = None) -> JSONResponse:
    """List reviews, optionally filtered by ``?status=``."""
    filter_status: ReviewStatus | None = None
    if status is not None:
        try:
            filter_status = ReviewStatus(status)
        except ValueError:
            return JSONResponse(
                {"error": f"Invalid status {status!r}. Valid: {[s.value for s in ReviewStatus]}"},
                status_code=400,
            )

    reviews = _store.list_reviews(status=filter_status)
    return JSONResponse([r.model_dump(mode="json") for r in reviews])

"""Tests for ReviewStore — mutable single-JSON review persistence."""

from __future__ import annotations

import json

import pytest

from vaig.core.review_store import ReviewStore
from vaig.skills.service_health.schema import FindingReview, ReportReview, ReviewStatus

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    """Return a ReviewStore backed by a temporary directory."""
    return ReviewStore(base_dir=tmp_path)


def _make_review(run_id: str = "run-1", **kwargs) -> ReportReview:
    return ReportReview(run_id=run_id, **kwargs)


# ── 1. save + read round-trip ─────────────────────────────────


def test_save_and_get_round_trip(store: ReviewStore) -> None:
    review = _make_review("abc-123", status=ReviewStatus.PENDING_REVIEW, reviewer="a@b.com")
    path = store.save(review)

    assert path.exists()
    loaded = store.get_by_run_id("abc-123")
    assert loaded is not None
    assert loaded.run_id == "abc-123"
    assert loaded.status == ReviewStatus.PENDING_REVIEW
    assert loaded.reviewer == "a@b.com"


# ── 2. update overwrites ─────────────────────────────────────


def test_update_overwrites(store: ReviewStore) -> None:
    review = _make_review("run-1", status=ReviewStatus.DRAFT)
    store.save(review)

    updated = _make_review("run-1", status=ReviewStatus.APPROVED, reviewer="admin@x.com")
    store.update("run-1", updated)

    loaded = store.get_by_run_id("run-1")
    assert loaded is not None
    assert loaded.status == ReviewStatus.APPROVED
    assert loaded.reviewer == "admin@x.com"


# ── 3. get_by_run_id returns None for missing ────────────────


def test_get_missing_returns_none(store: ReviewStore) -> None:
    assert store.get_by_run_id("nonexistent") is None


# ── 4. list_reviews filters by status ────────────────────────


def test_list_reviews_filters_by_status(store: ReviewStore) -> None:
    store.save(_make_review("r1", status=ReviewStatus.APPROVED))
    store.save(_make_review("r2", status=ReviewStatus.PENDING_REVIEW))
    store.save(_make_review("r3", status=ReviewStatus.PENDING_REVIEW))

    all_reviews = store.list_reviews()
    assert len(all_reviews) == 3

    pending = store.list_reviews(status=ReviewStatus.PENDING_REVIEW)
    assert len(pending) == 2
    assert all(r.status == ReviewStatus.PENDING_REVIEW for r in pending)


# ── 5. is_approved true / false ──────────────────────────────


def test_is_approved_true_false(store: ReviewStore) -> None:
    store.save(_make_review("approved-run", status=ReviewStatus.APPROVED))
    store.save(_make_review("pending-run", status=ReviewStatus.PENDING_REVIEW))

    assert store.is_approved("approved-run") is True
    assert store.is_approved("pending-run") is False
    assert store.is_approved("missing-run") is False


# ── 6. invalid run_id raises ValueError ──────────────────────


def test_invalid_run_id_raises(store: ReviewStore) -> None:
    with pytest.raises(ValueError, match="unsafe characters"):
        store.save(_make_review("../../etc/passwd"))

    with pytest.raises(ValueError, match="unsafe characters"):
        store.get_by_run_id("bad run id")


# ── 7. atomic write does not corrupt ─────────────────────────


def test_atomic_write_produces_valid_json(store: ReviewStore, tmp_path) -> None:
    review = _make_review(
        "atomic-test",
        status=ReviewStatus.APPROVED,
        finding_reviews=[FindingReview(finding_id="f1", status=ReviewStatus.APPROVED)],
        overall_comment="All good",
    )
    path = store.save(review)

    # Verify the raw file is valid JSON.
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["run_id"] == "atomic-test"
    assert parsed["status"] == "approved"
    assert len(parsed["finding_reviews"]) == 1


# ── 8. base_dir safety ───────────────────────────────────────


def test_base_dir_safety_rejects_system_path() -> None:
    from pathlib import Path

    with pytest.raises(ValueError, match="base_dir must be under"):
        ReviewStore(base_dir=Path("/etc/vaig-reviews"))

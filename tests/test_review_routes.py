"""Tests for review API routes (GET/POST /api/reviews)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from vaig.skills.service_health.schema import (
    ReportReview,
    ReviewStatus,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch):
    """Enable dev mode so is_admin returns True by default."""
    monkeypatch.setenv("VAIG_WEB_DEV_MODE", "true")


@pytest.fixture()
def tmp_store(tmp_path):
    """Return a ReviewStore backed by a temporary directory."""
    from vaig.core.review_store import ReviewStore

    return ReviewStore(base_dir=tmp_path)


@pytest.fixture()
def client(tmp_store):
    """Create a TestClient that patches the module-level _store."""
    from vaig.web.routes import reviews as reviews_mod

    original = reviews_mod._store
    reviews_mod._store = tmp_store
    try:
        from vaig.web.app import create_app

        app = create_app()
        yield TestClient(app)
    finally:
        reviews_mod._store = original


# ── Helpers ───────────────────────────────────────────────────


def _post_review(client: TestClient, run_id: str, **kwargs):
    body = {
        "status": kwargs.get("status", "pending_review"),
        "finding_reviews": kwargs.get("finding_reviews", []),
        "overall_comment": kwargs.get("overall_comment", ""),
    }
    return client.post(f"/api/reviews/{run_id}", json=body)


# ── 1. GET existing review ───────────────────────────────────


def test_get_existing_review(client: TestClient, tmp_store) -> None:
    tmp_store.save(
        ReportReview(run_id="r1", status=ReviewStatus.PENDING_REVIEW, reviewer="a@b.com")
    )
    resp = client.get("/api/reviews/r1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "r1"
    assert data["status"] == "pending_review"


# ── 2. GET 404 for missing ──────────────────────────────────


def test_get_missing_review_404(client: TestClient) -> None:
    resp = client.get("/api/reviews/nonexistent")
    assert resp.status_code == 404


# ── 3. POST create review ───────────────────────────────────


def test_post_create_review(client: TestClient) -> None:
    resp = _post_review(client, "new-run", status="approved", overall_comment="LGTM")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "new-run"
    assert data["status"] == "approved"
    assert data["overall_comment"] == "LGTM"


# ── 4. POST update review ───────────────────────────────────


def test_post_update_review(client: TestClient) -> None:
    _post_review(client, "upd-run", status="pending_review")
    resp = _post_review(client, "upd-run", status="approved", overall_comment="Updated")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    assert resp.json()["overall_comment"] == "Updated"


# ── 5. POST non-admin returns 403 ───────────────────────────


def test_post_non_admin_403(client: TestClient, monkeypatch) -> None:
    # Disable dev mode so is_admin actually checks admin list.
    monkeypatch.setenv("VAIG_WEB_DEV_MODE", "false")
    monkeypatch.setenv("VAIG_WEB_ADMIN_EMAILS", "admin@x.com")
    # IAP header for a non-admin user.
    resp = client.post(
        "/api/reviews/r1",
        json={"status": "approved"},
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:user@y.com"},
    )
    assert resp.status_code == 403


# ── 6. GET list all reviews ─────────────────────────────────


def test_list_all_reviews(client: TestClient, tmp_store) -> None:
    tmp_store.save(ReportReview(run_id="a", status=ReviewStatus.APPROVED))
    tmp_store.save(ReportReview(run_id="b", status=ReviewStatus.PENDING_REVIEW))
    resp = client.get("/api/reviews")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── 7. GET list filtered by status ──────────────────────────


def test_list_filtered_by_status(client: TestClient, tmp_store) -> None:
    tmp_store.save(ReportReview(run_id="x", status=ReviewStatus.APPROVED))
    tmp_store.save(ReportReview(run_id="y", status=ReviewStatus.PENDING_REVIEW))
    tmp_store.save(ReportReview(run_id="z", status=ReviewStatus.PENDING_REVIEW))

    resp = client.get("/api/reviews?status=pending_review")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(r["status"] == "pending_review" for r in data)

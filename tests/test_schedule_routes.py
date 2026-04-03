"""Tests for ``/portal/schedules`` web routes."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.core.scheduler import ScanResult, ScheduleInfo

# Ensure dev mode is enabled for tests (IAP bypass)
os.environ.setdefault("VAIG_WEB_DEV_MODE", "true")

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed — skipping schedule routes tests", allow_module_level=True)


# ── Fixtures ─────────────────────────────────────────────────


def _make_engine() -> MagicMock:
    """Build a mock SchedulerEngine."""
    engine = MagicMock()
    engine.list_schedules = AsyncMock(
        return_value=[
            ScheduleInfo(
                schedule_id="sched-1",
                cluster_name="prod-us",
                namespace="payments",
                interval_minutes=30,
                cron_expression=None,
                paused=False,
                next_fire_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            ),
        ],
    )
    engine.add_schedule = AsyncMock(return_value="sched-new-uuid")
    engine.remove_schedule = AsyncMock(return_value=True)
    engine.run_now = AsyncMock(
        return_value=ScanResult(
            id="run-1",
            schedule_id="sched-1",
            started_at="2026-06-01T12:00:00",
            completed_at="2026-06-01T12:01:00",
            status="success",
            report_json=None,
            diff_json=None,
            alerts_sent=1,
        ),
    )
    engine.get_history = AsyncMock(
        return_value=[
            ScanResult(
                id="run-1",
                schedule_id="sched-1",
                started_at="2026-06-01T12:00:00",
                completed_at="2026-06-01T12:01:00",
                status="success",
                report_json=None,
                diff_json=None,
                alerts_sent=1,
            ),
        ],
    )
    return engine


@pytest.fixture()
def client() -> TestClient:
    """Create a FastAPI TestClient with a mock scheduler engine."""
    from vaig.web.app import create_app

    app = create_app()

    # Register schedules router
    from vaig.web.routes.schedules import router as schedules_router

    app.include_router(schedules_router)

    # Attach mock engine
    app.state.scheduler_engine = _make_engine()

    return TestClient(app)


@pytest.fixture()
def client_no_engine() -> TestClient:
    """Create a TestClient WITHOUT a scheduler engine (503 expected)."""
    from vaig.web.app import create_app

    app = create_app()

    from vaig.web.routes.schedules import router as schedules_router

    app.include_router(schedules_router)

    # Ensure no engine attached
    app.state.scheduler_engine = None

    return TestClient(app)


# ── List schedules ───────────────────────────────────────────


class TestListSchedules:
    def test_list_json(self, client: TestClient) -> None:
        resp = client.get("/portal/schedules", headers={"accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["cluster_name"] == "prod-us"
        assert data[0]["schedule_id"] == "sched-1"

    def test_list_html(self, client: TestClient) -> None:
        resp = client.get("/portal/schedules", headers={"accept": "text/html"})
        assert resp.status_code == 200
        assert "prod-us" in resp.text


# ── Create schedule ──────────────────────────────────────────


class TestCreateSchedule:
    def test_create_success(self, client: TestClient) -> None:
        resp = client.post(
            "/portal/schedules",
            json={"cluster_name": "staging-eu", "interval_minutes": 60},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["schedule_id"] == "sched-new-uuid"
        assert data["status"] == "created"

    def test_create_missing_cluster(self, client: TestClient) -> None:
        resp = client.post("/portal/schedules", json={"namespace": "default"})
        assert resp.status_code == 400
        assert "cluster_name" in resp.json()["detail"]

    def test_create_with_cron(self, client: TestClient) -> None:
        resp = client.post(
            "/portal/schedules",
            json={"cluster_name": "prod-us", "cron": "0 */6 * * *"},
        )
        assert resp.status_code == 201


# ── Delete schedule ──────────────────────────────────────────


class TestDeleteSchedule:
    def test_delete_existing(self, client: TestClient) -> None:
        resp = client.delete("/portal/schedules/sched-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

    def test_delete_nonexistent(self, client: TestClient) -> None:
        engine = client.app.state.scheduler_engine  # type: ignore[union-attr]
        engine.remove_schedule = AsyncMock(return_value=False)
        resp = client.delete("/portal/schedules/nonexistent")
        assert resp.status_code == 404


# ── Trigger run-now ──────────────────────────────────────────


class TestTriggerSchedule:
    def test_trigger_success(self, client: TestClient) -> None:
        resp = client.post("/portal/schedules/sched-1/trigger")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["alerts_sent"] == 1

    def test_trigger_unknown(self, client: TestClient) -> None:
        engine = client.app.state.scheduler_engine  # type: ignore[union-attr]
        engine.run_now = AsyncMock(side_effect=ValueError("Unknown schedule_id: bad"))
        resp = client.post("/portal/schedules/bad/trigger")
        assert resp.status_code == 404


# ── History ──────────────────────────────────────────────────


class TestScheduleHistory:
    def test_history_returns_runs(self, client: TestClient) -> None:
        resp = client.get("/portal/schedules/sched-1/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "success"

    def test_history_empty(self, client: TestClient) -> None:
        engine = client.app.state.scheduler_engine  # type: ignore[union-attr]
        engine.get_history = AsyncMock(return_value=[])
        resp = client.get("/portal/schedules/sched-1/history")
        assert resp.status_code == 200
        assert resp.json() == []


# ── No engine (503) ──────────────────────────────────────────


class TestNoEngine:
    def test_list_503(self, client_no_engine: TestClient) -> None:
        resp = client_no_engine.get(
            "/portal/schedules",
            headers={"accept": "application/json"},
        )
        assert resp.status_code == 503

    def test_create_503(self, client_no_engine: TestClient) -> None:
        resp = client_no_engine.post(
            "/portal/schedules",
            json={"cluster_name": "prod"},
        )
        assert resp.status_code == 503

    def test_trigger_503(self, client_no_engine: TestClient) -> None:
        resp = client_no_engine.post("/portal/schedules/sched-1/trigger")
        assert resp.status_code == 503

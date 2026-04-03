"""Tests for ``vaig schedule`` CLI commands."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.commands.schedule import schedule_app
from vaig.core.config import ScheduleConfig, ScheduleTarget
from vaig.core.scheduler import ScanResult, ScheduleInfo

runner = CliRunner()


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Minimal Settings mock with schedule config."""
    settings = MagicMock()
    settings.schedule = ScheduleConfig(
        enabled=True,
        default_interval_minutes=30,
        targets=[ScheduleTarget(cluster_name="test-cluster")],
    )
    return settings


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Mock SchedulerEngine with async methods."""
    engine = MagicMock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.add_schedule = AsyncMock(return_value="sched-uuid-1234")
    engine.remove_schedule = AsyncMock(return_value=True)
    engine.run_now = AsyncMock(
        return_value=ScanResult(
            id="run-1",
            schedule_id="sched-uuid-1234",
            started_at="2026-01-01T00:00:00",
            completed_at="2026-01-01T00:01:00",
            status="success",
            report_json=None,
            diff_json=None,
            alerts_sent=0,
        ),
    )
    engine.list_schedules = AsyncMock(
        return_value=[
            ScheduleInfo(
                schedule_id="sched-uuid-1234",
                cluster_name="prod-us",
                namespace="payments",
                interval_minutes=30,
                cron_expression=None,
                paused=False,
                next_fire_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
        ],
    )
    return engine


def _patch_get_engine(mock_engine: MagicMock, mock_settings: MagicMock):
    """Return a patch context for ``_get_engine``."""
    return patch(
        "vaig.cli.commands.schedule._get_engine",
        return_value=(mock_engine, mock_settings),
    )


# ── add ──────────────────────────────────────────────────────


class TestScheduleAdd:
    def test_add_with_cluster(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["add", "--cluster", "prod-us"])
        assert result.exit_code == 0
        assert "sched-uuid-1234" in result.output
        mock_engine.add_schedule.assert_awaited_once()

    def test_add_with_interval(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(
                schedule_app,
                ["add", "--cluster", "prod-us", "--interval", "60"],
            )
        assert result.exit_code == 0
        call_kwargs = mock_engine.add_schedule.call_args
        assert call_kwargs.kwargs.get("interval_minutes") == 60 or call_kwargs[1].get("interval_minutes") == 60

    def test_add_with_cron(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(
                schedule_app,
                ["add", "--cluster", "prod-us", "--cron", "*/30 * * * *"],
            )
        assert result.exit_code == 0
        call_kwargs = mock_engine.add_schedule.call_args
        cron_val = call_kwargs.kwargs.get("cron") or call_kwargs[1].get("cron")
        assert cron_val == "*/30 * * * *"

    def test_add_with_namespace(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(
                schedule_app,
                ["add", "--cluster", "prod-us", "--namespace", "payments"],
            )
        assert result.exit_code == 0
        target = mock_engine.add_schedule.call_args[0][0]
        assert target.namespace == "payments"

    def test_add_with_all_namespaces(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(
                schedule_app,
                ["add", "--cluster", "prod-us", "--all-namespaces"],
            )
        assert result.exit_code == 0
        target = mock_engine.add_schedule.call_args[0][0]
        assert target.all_namespaces is True


# ── list ─────────────────────────────────────────────────────


class TestScheduleList:
    def test_list_with_schedules(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["list"])
        assert result.exit_code == 0
        assert "prod-us" in result.output
        assert "payments" in result.output

    def test_list_empty(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        mock_engine.list_schedules = AsyncMock(return_value=[])
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["list"])
        assert result.exit_code == 0
        assert "No schedules" in result.output


# ── remove ───────────────────────────────────────────────────


class TestScheduleRemove:
    def test_remove_existing(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["remove", "sched-uuid-1234"])
        assert result.exit_code == 0
        assert "removed" in result.output

    def test_remove_nonexistent(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        mock_engine.remove_schedule = AsyncMock(return_value=False)
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["remove", "nonexistent-id"])
        assert result.exit_code == 1
        assert "not found" in result.output


# ── run-now ──────────────────────────────────────────────────


class TestScheduleRunNow:
    def test_run_now_success(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["run-now", "sched-uuid-1234"])
        assert result.exit_code == 0
        assert "success" in result.output

    def test_run_now_unknown_schedule(
        self, mock_engine: MagicMock, mock_settings: MagicMock,
    ) -> None:
        mock_engine.run_now = AsyncMock(side_effect=ValueError("Unknown schedule_id: bad-id"))
        with _patch_get_engine(mock_engine, mock_settings):
            result = runner.invoke(schedule_app, ["run-now", "bad-id"])
        assert result.exit_code == 1
        assert "Unknown schedule_id" in result.output


# ── status ───────────────────────────────────────────────────


class TestScheduleStatus:
    def test_status_shows_config(self, mock_settings: MagicMock) -> None:
        with patch("vaig.cli.commands.schedule.typer"):  # noqa: SIM117
            with patch(
                "vaig.core.config.get_settings",
                return_value=mock_settings,
            ):
                result = runner.invoke(schedule_app, ["status"])
        assert result.exit_code == 0


# ── stop ─────────────────────────────────────────────────────


class TestScheduleStop:
    def test_stop_prints_message(self) -> None:
        result = runner.invoke(schedule_app, ["stop"])
        assert result.exit_code == 0
        assert "Ctrl+C" in result.output

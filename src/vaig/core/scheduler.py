"""Scheduler engine — runs headless health scans on cron/interval.

Uses APScheduler 4.x ``AsyncScheduler`` with ``MemoryDataStore`` (default)
or a caller-provided data store.  Scan results are diffed against the
previous run via :func:`~vaig.skills.service_health.diff.compute_report_diff`,
and alerts are dispatched only when new findings exceed the configured
severity threshold.

Budget enforcement (daily max analyses) and run history are persisted
in a SQLite database at ``settings.schedule.db_path``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from apscheduler import AsyncScheduler
from apscheduler.datastores.memory import MemoryDataStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from vaig.core.config import ScheduleTarget

if TYPE_CHECKING:
    from apscheduler.abc import DataStore

    from vaig.core.config import Settings
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

# ── Severity ordering (reused for threshold comparison) ──────

_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "INFO": 0,
}

# ── Result dataclasses ───────────────────────────────────────


@dataclass
class ScheduleInfo:
    """Summary of a registered schedule."""

    schedule_id: str
    cluster_name: str
    namespace: str
    interval_minutes: int | None
    cron_expression: str | None
    paused: bool
    next_fire_time: datetime | None


@dataclass
class ScanResult:
    """One row from the ``schedule_runs`` table."""

    id: str
    schedule_id: str
    started_at: str
    completed_at: str | None
    status: str
    report_json: str | None
    diff_json: str | None
    alerts_sent: int


@dataclass
class _ScheduleMeta:
    """In-memory metadata for a registered schedule target."""

    target: ScheduleTarget
    interval_minutes: int | None = None
    cron_expression: str | None = None
    run_number: int = 0


# ── SQLite DDL ───────────────────────────────────────────────

_DDL_SCHEDULE_RUNS = """
CREATE TABLE IF NOT EXISTS schedule_runs (
    id              TEXT PRIMARY KEY,
    schedule_id     TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    report_json     TEXT,
    diff_json       TEXT,
    alerts_sent     INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_BUDGET_USAGE = """
CREATE TABLE IF NOT EXISTS budget_usage (
    date            TEXT PRIMARY KEY,
    analysis_count  INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_INDEX_RUNS = """
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule_id
    ON schedule_runs (schedule_id, started_at DESC);
"""


# ── Engine ───────────────────────────────────────────────────


@dataclass
class _EngineState:
    """Mutable runtime state kept outside ``__init__`` to simplify testing."""

    previous_reports: dict[str, HealthReport] = field(default_factory=dict)
    schedule_meta: dict[str, _ScheduleMeta] = field(default_factory=dict)
    run_task: asyncio.Task[None] | None = None


class SchedulerEngine:
    """Drives periodic headless health scans with diff-based alerting.

    Usage::

        engine = SchedulerEngine(settings)
        await engine.start()
        sid = await engine.add_schedule(target, interval_minutes=30)
        # … later …
        await engine.stop()
    """

    def __init__(
        self,
        settings: Settings,
        *,
        data_store: DataStore | None = None,
    ) -> None:
        self._settings = settings
        self._cfg = settings.schedule
        self._db_path = Path(self._cfg.db_path).expanduser()
        self._data_store = data_store or MemoryDataStore()
        self._scheduler = AsyncScheduler(data_store=self._data_store)
        self._state = _EngineState()
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self, *, process: bool = True) -> None:
        """Enter the scheduler context and optionally start background processing.

        Args:
            process: When *True* (default), launch the background event loop
                that fires scheduled jobs.  Pass *False* in tests to
                initialise services without the ``anyio`` task group that
                ``run_until_stopped()`` requires.
        """
        if self._started:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._init_db()
        await self._scheduler.__aenter__()
        if process:
            self._state.run_task = asyncio.create_task(
                self._scheduler.run_until_stopped(),
                name="vaig-scheduler",
            )
        self._started = True
        logger.info("Scheduler engine started (db=%s, process=%s)", self._db_path, process)

    async def stop(self) -> None:
        """Stop background processing and exit the scheduler context."""
        if not self._started:
            return
        if self._state.run_task is not None:
            await self._scheduler.stop()
            try:
                await self._state.run_task
            except asyncio.CancelledError:
                pass
            self._state.run_task = None
        await self._scheduler.__aexit__(None, None, None)
        self._started = False
        logger.info("Scheduler engine stopped")

    # ── Schedule management ──────────────────────────────────

    async def add_schedule(
        self,
        target: ScheduleTarget,
        interval_minutes: int | None = None,
        cron: str | None = None,
    ) -> str:
        """Register a recurring health-scan schedule.

        Returns the generated ``schedule_id``.
        """
        schedule_id = str(uuid.uuid4())

        if cron:
            trigger: IntervalTrigger | CronTrigger = CronTrigger.from_crontab(cron)
        else:
            minutes = interval_minutes or self._cfg.default_interval_minutes
            trigger = IntervalTrigger(minutes=minutes)

        await self._scheduler.add_schedule(
            self._scan_job,
            trigger,
            id=schedule_id,
            kwargs={"schedule_id": schedule_id},
            misfire_grace_time=self._cfg.misfire_grace_time,
        )

        self._state.schedule_meta[schedule_id] = _ScheduleMeta(
            target=target,
            interval_minutes=interval_minutes or (None if cron else self._cfg.default_interval_minutes),
            cron_expression=cron,
        )

        logger.info(
            "Schedule added: id=%s, cluster=%s, ns=%s",
            schedule_id,
            target.cluster_name,
            target.namespace or "(all)",
        )
        return schedule_id

    async def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule. Returns ``True`` if it existed."""
        if schedule_id not in self._state.schedule_meta:
            return False
        try:
            await self._scheduler.remove_schedule(schedule_id)
        except Exception:  # noqa: BLE001
            logger.warning("Schedule %s not found in APScheduler", schedule_id)
        self._state.schedule_meta.pop(schedule_id, None)
        self._state.previous_reports.pop(schedule_id, None)
        logger.info("Schedule removed: %s", schedule_id)
        return True

    async def list_schedules(self) -> list[ScheduleInfo]:
        """Return metadata for all registered schedules."""
        ap_schedules = await self._scheduler.get_schedules()
        results: list[ScheduleInfo] = []
        for sched in ap_schedules:
            meta = self._state.schedule_meta.get(sched.id)
            results.append(
                ScheduleInfo(
                    schedule_id=sched.id,
                    cluster_name=meta.target.cluster_name if meta else "",
                    namespace=meta.target.namespace if meta else "",
                    interval_minutes=meta.interval_minutes if meta else None,
                    cron_expression=meta.cron_expression if meta else None,
                    paused=sched.paused,
                    next_fire_time=sched.next_fire_time,
                ),
            )
        return results

    async def run_now(self, schedule_id: str) -> ScanResult:
        """Trigger a scan immediately for an existing schedule."""
        meta = self._state.schedule_meta.get(schedule_id)
        if meta is None:
            msg = f"Unknown schedule_id: {schedule_id}"
            raise ValueError(msg)
        return await self._scan_job(schedule_id=schedule_id)

    async def get_history(
        self,
        schedule_id: str,
        limit: int = 20,
    ) -> list[ScanResult]:
        """Return recent scan results for a schedule from SQLite."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM schedule_runs WHERE schedule_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (schedule_id, limit),
            )
            rows = await cursor.fetchall()
        return [
            ScanResult(
                id=row["id"],
                schedule_id=row["schedule_id"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                status=row["status"],
                report_json=row["report_json"],
                diff_json=row["diff_json"],
                alerts_sent=row["alerts_sent"],
            )
            for row in rows
        ]

    # ── Scan job (APScheduler callback) ──────────────────────

    async def _scan_job(self, *, schedule_id: str) -> ScanResult:
        """Execute one health scan — called by APScheduler or ``run_now``.

        Flow:
        1. Check daily budget → skip if exhausted.
        2. Run ``execute_skill_headless()`` → get ``OrchestratorResult``.
        3. Diff against previous report → conditional alert.
        4. Persist run to ``schedule_runs`` + update budget counter.
        5. Optionally export via ``auto_export_report()``.
        """
        meta = self._state.schedule_meta.get(schedule_id)
        if meta is None:
            logger.error("_scan_job called with unknown schedule_id=%s", schedule_id)
            return self._make_result(schedule_id, "error", error="unknown_schedule_id")

        meta.run_number += 1
        run_id = str(uuid.uuid4())
        started_at = datetime.now(UTC).isoformat()

        # ── 1. Budget check ──────────────────────────────────
        if not await self._check_budget():
            logger.warning(
                "Daily budget exhausted (%d/%d) — skipping schedule %s",
                await self._get_daily_count(),
                self._cfg.daily_max_analyses,
                schedule_id,
            )
            return await self._record_run(
                run_id, schedule_id, started_at, "skipped_budget",
            )

        # ── 2. Execute headless scan ─────────────────────────
        try:
            from vaig.core.config import GKEConfig
            from vaig.core.headless import execute_skill_headless
            from vaig.skills.discovery.skill import DiscoverySkill

            target = meta.target
            gke_config = GKEConfig(
                cluster_name=target.cluster_name,
                default_namespace=target.namespace or "default",
                project_id=self._settings.gcp.project_id,
                location=self._settings.gcp.location,
            )
            skill = DiscoverySkill()
            query = self._build_query(target)

            orch_result = execute_skill_headless(
                self._settings,
                skill,
                query,
                gke_config,
            )
            report: HealthReport | None = orch_result.structured_report
        except Exception:
            logger.exception("Headless scan failed for schedule %s", schedule_id)
            await self._increment_budget()
            return await self._record_run(
                run_id, schedule_id, started_at, "error",
            )

        if report is None:
            logger.warning("No structured report from schedule %s", schedule_id)
            await self._increment_budget()
            return await self._record_run(
                run_id, schedule_id, started_at, "no_report",
            )

        # ── 3. Diff + conditional alert ──────────────────────
        from vaig.skills.service_health.diff import compute_report_diff

        alerts_sent = 0
        diff_data: dict[str, Any] | None = None
        previous = self._state.previous_reports.get(schedule_id)

        if previous is not None:
            diff = compute_report_diff(report, previous)
            diff_data = {
                "has_changes": diff.has_changes,
                "summary_line": diff.summary_line,
                "new_findings": len(diff.new_findings),
                "resolved_findings": len(diff.resolved_findings),
            }

            if diff.has_changes and self._has_alertable_findings(diff):
                alerts_sent = self._dispatch_alert(
                    report, meta, diff,
                )

        # Update previous-report cache
        self._state.previous_reports[schedule_id] = report
        await self._increment_budget()

        # ── 4. Persist run ───────────────────────────────────
        report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
        result = await self._record_run(
            run_id,
            schedule_id,
            started_at,
            "success",
            report_json=json.dumps(report_dict),
            diff_json=json.dumps(diff_data) if diff_data else None,
            alerts_sent=alerts_sent,
        )

        # ── 5. Auto-export ───────────────────────────────────
        if self._cfg.store_results and self._settings.export.enabled:
            try:
                from vaig.core.export import auto_export_report

                tagged_run_id = f"scheduled:{schedule_id}:{meta.run_number}"
                auto_export_report(
                    self._settings.export,
                    report,
                    run_id=tagged_run_id,
                    cluster_name=meta.target.cluster_name,
                    namespace=meta.target.namespace,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Auto-export failed for run %s", run_id, exc_info=True)

        logger.info(
            "Scan complete: schedule=%s, run=%d, status=success, alerts=%d",
            schedule_id,
            meta.run_number,
            alerts_sent,
        )
        return result

    # ── Alert helpers ────────────────────────────────────────

    def _has_alertable_findings(self, diff: Any) -> bool:
        """Check if diff contains new findings at or above the threshold."""
        threshold = _SEVERITY_ORDER.get(self._cfg.alert_severity_threshold.upper(), 0)
        for finding in diff.new_findings:
            severity_val = _SEVERITY_ORDER.get(finding.severity.value.upper(), 0)
            if severity_val >= threshold:
                return True
        return False

    def _dispatch_alert(
        self,
        report: HealthReport,
        meta: _ScheduleMeta,
        diff: Any,
    ) -> int:
        """Send alert via NotificationDispatcher. Returns count of alerts sent."""
        try:
            from vaig.integrations.dispatcher import (
                AlertContext,
                NotificationDispatcher,
            )

            dispatcher = NotificationDispatcher.from_config(self._settings)
            alert_context = AlertContext(
                alert_id=f"sched-{meta.target.cluster_name}-{meta.run_number}",
                source="scheduler",
                service_name=meta.target.cluster_name,
                cluster_name=meta.target.cluster_name,
                namespace=meta.target.namespace,
                schedule_id=next(
                    (sid for sid, m in self._state.schedule_meta.items() if m is meta),
                    "",
                ),
                run_number=meta.run_number,
                is_scheduled=True,
            )
            result = dispatcher.dispatch(report, alert_context)
            sent = int(not result.has_errors)
            if result.has_errors:
                logger.warning("Alert dispatch had errors: %s", result.errors)
            return sent
        except Exception:  # noqa: BLE001
            logger.exception("Failed to dispatch alert")
            return 0

    @staticmethod
    def _build_query(target: ScheduleTarget) -> str:
        """Build the investigation query string for a target."""
        parts = [f"Run a full health check on cluster {target.cluster_name}"]
        if target.namespace:
            parts.append(f"namespace {target.namespace}")
        if target.all_namespaces:
            parts.append("across all namespaces")
        if target.skip_healthy:
            parts.append("— focus on unhealthy services")
        return ", ".join(parts) + "."

    @staticmethod
    def _make_result(
        schedule_id: str,
        status: str,
        *,
        error: str = "",
    ) -> ScanResult:
        """Create a minimal ScanResult for error/skip paths."""
        now = datetime.now(UTC).isoformat()
        return ScanResult(
            id=str(uuid.uuid4()),
            schedule_id=schedule_id,
            started_at=now,
            completed_at=now,
            status=status,
            report_json=None,
            diff_json=json.dumps({"error": error}) if error else None,
            alerts_sent=0,
        )

    # ── SQLite helpers ───────────────────────────────────────

    async def _init_db(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_DDL_SCHEDULE_RUNS)
            await db.execute(_DDL_BUDGET_USAGE)
            await db.execute(_DDL_INDEX_RUNS)
            await db.commit()

    async def _check_budget(self) -> bool:
        """Return ``True`` if another analysis is allowed today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT analysis_count FROM budget_usage WHERE date = ?",
                (today,),
            )
            row = await cursor.fetchone()
        current = row[0] if row else 0
        return current < self._cfg.daily_max_analyses

    async def _get_daily_count(self) -> int:
        """Return today's analysis count."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT analysis_count FROM budget_usage WHERE date = ?",
                (today,),
            )
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def _increment_budget(self) -> None:
        """Increment today's analysis counter (insert-or-update)."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO budget_usage (date, analysis_count) VALUES (?, 1) "
                "ON CONFLICT(date) DO UPDATE SET analysis_count = analysis_count + 1",
                (today,),
            )
            await db.commit()

    async def _record_run(
        self,
        run_id: str,
        schedule_id: str,
        started_at: str,
        status: str,
        *,
        report_json: str | None = None,
        diff_json: str | None = None,
        alerts_sent: int = 0,
    ) -> ScanResult:
        """Insert a row into ``schedule_runs`` and return the result."""
        completed_at = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO schedule_runs "
                "(id, schedule_id, started_at, completed_at, status, "
                " report_json, diff_json, alerts_sent) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    schedule_id,
                    started_at,
                    completed_at,
                    status,
                    report_json,
                    diff_json,
                    alerts_sent,
                ),
            )
            await db.commit()
        return ScanResult(
            id=run_id,
            schedule_id=schedule_id,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            report_json=report_json,
            diff_json=diff_json,
            alerts_sent=alerts_sent,
        )

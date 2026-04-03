"""Tests for SchedulerEngine — lifecycle, budget, diff alerting, history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from apscheduler.datastores.memory import MemoryDataStore

from vaig.core.config import ScheduleConfig, ScheduleTarget, Settings
from vaig.core.scheduler import _SEVERITY_ORDER, SchedulerEngine

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary SQLite DB path."""
    return tmp_path / "test_scheduler.db"


@pytest.fixture()
def schedule_target() -> ScheduleTarget:
    return ScheduleTarget(cluster_name="test-cluster", namespace="default")


@pytest.fixture()
def make_settings(tmp_db: Path) -> Settings:
    """Create settings with a temp DB path and scheduling enabled."""
    s = Settings()
    s.schedule = ScheduleConfig(
        enabled=True,
        default_interval_minutes=30,
        daily_max_analyses=10,
        db_path=str(tmp_db),
        targets=[ScheduleTarget(cluster_name="test-cluster")],
    )
    return s


@pytest.fixture()
def engine(make_settings: Settings) -> SchedulerEngine:
    """Create a SchedulerEngine with MemoryDataStore for testing."""
    return SchedulerEngine(make_settings, data_store=MemoryDataStore())


# ── Mock health report helpers ───────────────────────────────


@dataclass
class _MockFinding:
    id: str
    title: str
    severity: Any  # uses .value

    @dataclass
    class _Sev:
        value: str

    @classmethod
    def create(cls, fid: str, title: str, severity: str) -> _MockFinding:
        return cls(id=fid, title=title, severity=cls._Sev(value=severity))

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "severity": self.severity.value}


@dataclass
class _MockExecSummary:
    overall_status: Any
    issues_found: int = 2
    critical_count: int = 1
    warning_count: int = 1
    scope: str = "test-cluster"
    summary_text: str = "Test summary"

    @dataclass
    class _Status:
        value: str

    @classmethod
    def create(cls, status: str = "DEGRADED") -> _MockExecSummary:
        return cls(overall_status=cls._Status(value=status))


@dataclass
class _MockReport:
    findings: list[_MockFinding] = field(default_factory=list)
    executive_summary: _MockExecSummary = field(
        default_factory=lambda: _MockExecSummary.create()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.model_dump() for f in self.findings],
            "executive_summary": {
                "overall_status": self.executive_summary.overall_status.value,
            },
        }


@dataclass
class _MockOrchestratorResult:
    structured_report: _MockReport | None = None
    success: bool = True
    run_cost_usd: float = 0.01


def _make_report(*findings_specs: tuple[str, str, str]) -> _MockReport:
    """Create a mock report with (id, title, severity) tuples."""
    findings = [
        _MockFinding.create(fid, title, sev)
        for fid, title, sev in findings_specs
    ]
    return _MockReport(findings=findings)


# ── Test: Engine lifecycle ───────────────────────────────────


class TestEngineLifecycle:
    @pytest.mark.asyncio()
    async def test_start_creates_db_tables(
        self, engine: SchedulerEngine, tmp_db: Path
    ) -> None:
        await engine.start(process=False)
        try:
            assert tmp_db.exists()
            async with aiosqlite.connect(str(tmp_db)) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in await cursor.fetchall()}
            assert "schedule_runs" in tables
            assert "budget_usage" in tables
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_start_stop_idempotent(self, engine: SchedulerEngine) -> None:
        await engine.start(process=False)
        await engine.start(process=False)  # should be no-op
        assert engine._started is True

        await engine.stop()
        await engine.stop()  # should be no-op
        assert engine._started is False

    @pytest.mark.asyncio()
    async def test_add_and_list_schedule(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=15)
            assert sid  # non-empty string

            schedules = await engine.list_schedules()
            assert len(schedules) == 1
            assert schedules[0].schedule_id == sid
            assert schedules[0].cluster_name == "test-cluster"
            assert schedules[0].namespace == "default"
            assert schedules[0].interval_minutes == 15
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_remove_schedule(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)
            assert await engine.remove_schedule(sid) is True
            assert await engine.list_schedules() == []
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_remove_nonexistent_returns_false(
        self, engine: SchedulerEngine
    ) -> None:
        await engine.start(process=False)
        try:
            assert await engine.remove_schedule("nonexistent-id") is False
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_add_schedule_with_cron(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(
                schedule_target, cron="0 */2 * * *"
            )
            schedules = await engine.list_schedules()
            assert len(schedules) == 1
            assert schedules[0].cron_expression == "0 */2 * * *"
        finally:
            await engine.stop()


# ── Test: Budget enforcement ─────────────────────────────────


class TestBudgetEnforcement:
    @pytest.mark.asyncio()
    async def test_budget_allows_when_under_limit(
        self, engine: SchedulerEngine
    ) -> None:
        await engine.start(process=False)
        try:
            assert await engine._check_budget() is True
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_budget_blocks_when_at_limit(
        self, engine: SchedulerEngine, tmp_db: Path
    ) -> None:
        await engine.start(process=False)
        try:
            # Manually set budget to the max (10)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            async with aiosqlite.connect(str(tmp_db)) as db:
                await db.execute(
                    "INSERT INTO budget_usage (date, analysis_count) VALUES (?, ?)",
                    (today, 10),
                )
                await db.commit()
            assert await engine._check_budget() is False
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_increment_budget(
        self, engine: SchedulerEngine, tmp_db: Path
    ) -> None:
        await engine.start(process=False)
        try:
            await engine._increment_budget()
            await engine._increment_budget()
            count = await engine._get_daily_count()
            assert count == 2
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_budget_exhausted_skips_scan(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget, tmp_db: Path
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            # Exhaust budget
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            async with aiosqlite.connect(str(tmp_db)) as db:
                await db.execute(
                    "INSERT INTO budget_usage (date, analysis_count) VALUES (?, ?)",
                    (today, 10),
                )
                await db.commit()

            result = await engine._scan_job(schedule_id=sid)
            assert result.status == "skipped_budget"
        finally:
            await engine.stop()


# ── Test: Diff-based alerting ────────────────────────────────


class TestDiffAlerting:
    @pytest.mark.asyncio()
    async def test_new_critical_finding_triggers_alert(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        """When new critical findings appear, an alert should be dispatched."""
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            # Set up previous report with findings A, B
            prev_report = _make_report(
                ("finding-a", "Finding A", "HIGH"),
                ("finding-b", "Finding B", "MEDIUM"),
            )
            engine._state.previous_reports[sid] = prev_report

            # New report adds finding C (CRITICAL)
            new_report = _make_report(
                ("finding-a", "Finding A", "HIGH"),
                ("finding-b", "Finding B", "MEDIUM"),
                ("finding-c", "Finding C", "CRITICAL"),
            )
            orch_result = _MockOrchestratorResult(structured_report=new_report)

            with (
                patch(
                    "vaig.core.headless.execute_skill_headless",
                    return_value=orch_result,
                ),
                patch(
                    "vaig.skills.discovery.skill.DiscoverySkill",
                    return_value=MagicMock(),
                ),
                patch.object(
                    engine, "_dispatch_alert", new_callable=AsyncMock, return_value=1
                ) as mock_dispatch,
            ):
                result = await engine._scan_job(schedule_id=sid)

            assert result.status == "success"
            assert result.alerts_sent == 1
            mock_dispatch.assert_called_once()
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_no_new_findings_suppresses_alert(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        """When findings are identical, no alert should be dispatched."""
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            # Previous and current have the same findings
            report = _make_report(
                ("finding-a", "Finding A", "HIGH"),
                ("finding-b", "Finding B", "MEDIUM"),
            )
            engine._state.previous_reports[sid] = report

            orch_result = _MockOrchestratorResult(structured_report=report)

            with (
                patch(
                    "vaig.core.headless.execute_skill_headless",
                    return_value=orch_result,
                ),
                patch(
                    "vaig.skills.discovery.skill.DiscoverySkill",
                    return_value=MagicMock(),
                ),
                patch.object(
                    engine, "_dispatch_alert", new_callable=AsyncMock, return_value=0
                ) as mock_dispatch,
            ):
                result = await engine._scan_job(schedule_id=sid)

            assert result.status == "success"
            assert result.alerts_sent == 0
            # dispatch should NOT be called when there's no change
            mock_dispatch.assert_not_called()
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_first_scan_no_diff_no_alert(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        """First scan has no previous report, so no diff and no alert."""
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            report = _make_report(("finding-a", "Finding A", "CRITICAL"))
            orch_result = _MockOrchestratorResult(structured_report=report)

            with (
                patch(
                    "vaig.core.headless.execute_skill_headless",
                    return_value=orch_result,
                ),
                patch(
                    "vaig.skills.discovery.skill.DiscoverySkill",
                    return_value=MagicMock(),
                ),
                patch.object(
                    engine, "_dispatch_alert", new_callable=AsyncMock, return_value=1
                ) as mock_dispatch,
            ):
                result = await engine._scan_job(schedule_id=sid)

            assert result.status == "success"
            # No previous report → no diff → no alert
            mock_dispatch.assert_not_called()
            # But report is cached for next run
            assert sid in engine._state.previous_reports
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_new_low_finding_below_threshold_no_alert(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        """New findings below threshold should not trigger alert."""
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            # Previous: finding A (HIGH)
            prev_report = _make_report(("finding-a", "Finding A", "HIGH"))
            engine._state.previous_reports[sid] = prev_report

            # New: finding A + finding B (LOW — below HIGH threshold)
            new_report = _make_report(
                ("finding-a", "Finding A", "HIGH"),
                ("finding-b", "Finding B", "LOW"),
            )
            orch_result = _MockOrchestratorResult(structured_report=new_report)

            with (
                patch(
                    "vaig.core.headless.execute_skill_headless",
                    return_value=orch_result,
                ),
                patch(
                    "vaig.skills.discovery.skill.DiscoverySkill",
                    return_value=MagicMock(),
                ),
                patch.object(
                    engine, "_dispatch_alert", new_callable=AsyncMock, return_value=0
                ) as mock_dispatch,
            ):
                result = await engine._scan_job(schedule_id=sid)

            assert result.status == "success"
            # Diff has changes (new finding B), but severity is LOW < HIGH threshold
            mock_dispatch.assert_not_called()
        finally:
            await engine.stop()


# ── Test: has_alertable_findings ─────────────────────────────


class TestHasAlertableFindings:
    def test_critical_above_high_threshold(self, engine: SchedulerEngine) -> None:
        diff = MagicMock()
        diff.new_findings = [_MockFinding.create("f1", "F1", "CRITICAL")]
        assert engine._has_alertable_findings(diff) is True

    def test_low_below_high_threshold(self, engine: SchedulerEngine) -> None:
        diff = MagicMock()
        diff.new_findings = [_MockFinding.create("f1", "F1", "LOW")]
        assert engine._has_alertable_findings(diff) is False

    def test_empty_new_findings(self, engine: SchedulerEngine) -> None:
        diff = MagicMock()
        diff.new_findings = []
        assert engine._has_alertable_findings(diff) is False


# ── Test: History retrieval ──────────────────────────────────


class TestHistory:
    @pytest.mark.asyncio()
    async def test_get_history_returns_runs(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)

            # Insert a run directly
            await engine._record_run(
                "run-1", sid, "2026-01-01T00:00:00", "success",
                report_json='{"test": true}',
            )
            await engine._record_run(
                "run-2", sid, "2026-01-01T01:00:00", "success",
            )

            history = await engine.get_history(sid, limit=10)
            assert len(history) == 2
            # Most recent first
            assert history[0].id == "run-2"
            assert history[1].id == "run-1"
            assert history[1].report_json == '{"test": true}'
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_get_history_empty(
        self, engine: SchedulerEngine
    ) -> None:
        await engine.start(process=False)
        try:
            history = await engine.get_history("nonexistent", limit=10)
            assert history == []
        finally:
            await engine.stop()


# ── Test: run_now ────────────────────────────────────────────


class TestRunNow:
    @pytest.mark.asyncio()
    async def test_run_now_unknown_schedule_raises(
        self, engine: SchedulerEngine
    ) -> None:
        await engine.start(process=False)
        try:
            with pytest.raises(ValueError, match="Unknown schedule_id"):
                await engine.run_now("nonexistent")
        finally:
            await engine.stop()

    @pytest.mark.asyncio()
    async def test_run_now_executes_scan(
        self, engine: SchedulerEngine, schedule_target: ScheduleTarget
    ) -> None:
        await engine.start(process=False)
        try:
            sid = await engine.add_schedule(schedule_target, interval_minutes=30)
            report = _make_report(("f1", "Finding 1", "HIGH"))
            orch_result = _MockOrchestratorResult(structured_report=report)

            with (
                patch(
                    "vaig.core.headless.execute_skill_headless",
                    return_value=orch_result,
                ),
                patch(
                    "vaig.skills.discovery.skill.DiscoverySkill",
                    return_value=MagicMock(),
                ),
            ):
                result = await engine.run_now(sid)

            assert result.status == "success"
        finally:
            await engine.stop()


# ── Test: Build query ────────────────────────────────────────


class TestBuildQuery:
    def test_basic_query(self) -> None:
        target = ScheduleTarget(cluster_name="prod-us")
        q = SchedulerEngine._build_query(target)
        assert "prod-us" in q

    def test_query_with_namespace(self) -> None:
        target = ScheduleTarget(cluster_name="prod-us", namespace="payments")
        q = SchedulerEngine._build_query(target)
        assert "payments" in q
        assert "prod-us" in q

    def test_query_with_all_namespaces(self) -> None:
        target = ScheduleTarget(
            cluster_name="prod-us", all_namespaces=True
        )
        q = SchedulerEngine._build_query(target)
        assert "all namespaces" in q


# ── Test: Severity ordering ──────────────────────────────────


class TestSeverityOrder:
    def test_severity_values(self) -> None:
        assert _SEVERITY_ORDER["CRITICAL"] > _SEVERITY_ORDER["HIGH"]
        assert _SEVERITY_ORDER["HIGH"] > _SEVERITY_ORDER["MEDIUM"]
        assert _SEVERITY_ORDER["MEDIUM"] > _SEVERITY_ORDER["LOW"]
        assert _SEVERITY_ORDER["LOW"] > _SEVERITY_ORDER["INFO"]

    def test_all_severities_present(self) -> None:
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        assert set(_SEVERITY_ORDER.keys()) == expected


# ── Test: State persistence across restarts ──────────────────


class TestStatePersistence:
    @pytest.mark.asyncio()
    async def test_schedule_meta_survives_restart(
        self, make_settings: Settings, schedule_target: ScheduleTarget
    ) -> None:
        """schedule_meta should be reloaded from SQLite on restart."""
        engine1 = SchedulerEngine(make_settings, data_store=MemoryDataStore())
        await engine1.start(process=False)
        try:
            sid = await engine1.add_schedule(schedule_target, interval_minutes=15)
        finally:
            await engine1.stop()

        # New engine instance — simulates process restart
        engine2 = SchedulerEngine(make_settings, data_store=MemoryDataStore())
        await engine2.start(process=False)
        try:
            assert sid in engine2._state.schedule_meta
            meta = engine2._state.schedule_meta[sid]
            assert meta.target.cluster_name == "test-cluster"
            assert meta.interval_minutes == 15
        finally:
            await engine2.stop()

    @pytest.mark.asyncio()
    async def test_remove_schedule_clears_persisted_state(
        self, make_settings: Settings, schedule_target: ScheduleTarget
    ) -> None:
        """Removing a schedule should remove it from SQLite too."""
        engine1 = SchedulerEngine(make_settings, data_store=MemoryDataStore())
        await engine1.start(process=False)
        try:
            sid = await engine1.add_schedule(schedule_target, interval_minutes=30)
            await engine1.remove_schedule(sid)
        finally:
            await engine1.stop()

        engine2 = SchedulerEngine(make_settings, data_store=MemoryDataStore())
        await engine2.start(process=False)
        try:
            assert sid not in engine2._state.schedule_meta
        finally:
            await engine2.stop()

    @pytest.mark.asyncio()
    async def test_db_connection_closed_on_stop(
        self, engine: SchedulerEngine
    ) -> None:
        """Persistent DB connection should be closed after stop()."""
        await engine.start(process=False)
        assert engine._state.db is not None
        await engine.stop()
        assert engine._state.db is None

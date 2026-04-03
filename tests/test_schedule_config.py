"""Tests for ScheduleConfig, ScheduleTarget, and AlertContext extensions."""

from __future__ import annotations

from vaig.core.config import ScheduleConfig, ScheduleTarget, Settings
from vaig.integrations.dispatcher import AlertContext


class TestScheduleTarget:
    def test_defaults(self) -> None:
        target = ScheduleTarget(cluster_name="prod-us")
        assert target.cluster_name == "prod-us"
        assert target.namespace == ""
        assert target.all_namespaces is False
        assert target.skip_healthy is True

    def test_full_config(self) -> None:
        target = ScheduleTarget(
            cluster_name="prod-eu",
            namespace="payments",
            all_namespaces=False,
            skip_healthy=False,
        )
        assert target.cluster_name == "prod-eu"
        assert target.namespace == "payments"
        assert target.skip_healthy is False


class TestScheduleConfig:
    def test_defaults(self) -> None:
        cfg = ScheduleConfig()
        assert cfg.enabled is False
        assert cfg.default_interval_minutes == 30
        assert cfg.cron_expression is None
        assert cfg.targets == []
        assert cfg.alert_severity_threshold == "HIGH"
        assert cfg.daily_max_analyses == 48
        assert cfg.per_schedule_max_analyses is None
        assert cfg.max_concurrent_scans == 1
        assert cfg.store_results is True
        assert cfg.misfire_grace_time == 900
        assert cfg.db_path == "~/.vaig/scheduler.db"

    def test_auto_enable_with_targets(self) -> None:
        cfg = ScheduleConfig(
            targets=[ScheduleTarget(cluster_name="prod-us")],
        )
        assert cfg.enabled is True

    def test_auto_enable_skipped_when_explicit_enabled(self) -> None:
        cfg = ScheduleConfig(enabled=True, targets=[])
        assert cfg.enabled is True

    def test_auto_enable_not_triggered_without_targets(self) -> None:
        cfg = ScheduleConfig(enabled=False, targets=[])
        assert cfg.enabled is False

    def test_multiple_targets(self) -> None:
        cfg = ScheduleConfig(
            targets=[
                ScheduleTarget(cluster_name="prod-us"),
                ScheduleTarget(cluster_name="prod-eu", namespace="payments"),
            ],
        )
        assert cfg.enabled is True
        assert len(cfg.targets) == 2

    def test_custom_fields(self) -> None:
        cfg = ScheduleConfig(
            default_interval_minutes=60,
            cron_expression="0 */2 * * *",
            daily_max_analyses=100,
            per_schedule_max_analyses=10,
            max_concurrent_scans=3,
            db_path="/tmp/test.db",
        )
        assert cfg.default_interval_minutes == 60
        assert cfg.cron_expression == "0 */2 * * *"
        assert cfg.daily_max_analyses == 100
        assert cfg.per_schedule_max_analyses == 10
        assert cfg.max_concurrent_scans == 3
        assert cfg.db_path == "/tmp/test.db"


class TestScheduleInSettings:
    def test_settings_has_schedule_field(self) -> None:
        s = Settings()
        assert hasattr(s, "schedule")
        assert isinstance(s.schedule, ScheduleConfig)

    def test_settings_schedule_defaults(self) -> None:
        s = Settings()
        assert s.schedule.enabled is False
        assert s.schedule.targets == []


class TestAlertContextExtension:
    def test_manual_dispatch_unchanged(self) -> None:
        ctx = AlertContext(
            alert_id="test-123",
            source="datadog",
            service_name="my-service",
        )
        assert ctx.schedule_id is None
        assert ctx.run_number is None
        assert ctx.is_scheduled is False

    def test_scheduled_dispatch(self) -> None:
        ctx = AlertContext(
            alert_id="sched-456",
            source="scheduler",
            service_name="payments",
            schedule_id="abc-123",
            run_number=42,
            is_scheduled=True,
        )
        assert ctx.schedule_id == "abc-123"
        assert ctx.run_number == 42
        assert ctx.is_scheduled is True

    def test_existing_fields_preserved(self) -> None:
        ctx = AlertContext(
            alert_id="dd-789",
            source="datadog",
            service_name="auth-svc",
            cluster_name="prod-us",
            namespace="auth",
        )
        assert ctx.alert_id == "dd-789"
        assert ctx.source == "datadog"
        assert ctx.service_name == "auth-svc"
        assert ctx.cluster_name == "prod-us"
        assert ctx.namespace == "auth"

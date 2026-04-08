"""Tests for the remediate CLI command — vaig remediate."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.core.config import RemediationConfig, Settings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def cli_app():
    """Create a fresh Typer app with the remediate command registered."""
    import typer

    from vaig.cli.commands.remediate import register

    test_app = typer.Typer()
    register(test_app)
    return test_app


@pytest.fixture()
def runner():
    """Typer CLI test runner."""
    return CliRunner()


def _make_settings(*, enabled: bool = True, **rem_kwargs: Any) -> Settings:
    """Build a Settings object with remediation configuration."""
    rem_config = RemediationConfig(enabled=enabled, **rem_kwargs)
    settings = Settings()
    settings.remediation = rem_config
    return settings


def _make_report_dict(
    *,
    recommendations: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal health report dict.

    Uses the same structure as HealthReport.model_validate expects.
    """
    if findings is None:
        findings = [
            {
                "id": "crashloop-payment-svc",
                "title": "CrashLoopBackOff in payment-svc",
                "severity": "critical",
                "category": "workload",
                "description": "Pod is crash-looping",
                "impact": "Service degraded",
                "evidence": ["restart count: 15"],
            },
        ]
    if recommendations is None:
        recommendations = [
            {
                "priority": 1,
                "title": "Restart payment-svc deployment",
                "description": "Rolling restart to recover from crashloop",
                "urgency": "high",
                "effort": "low",
                "command": "kubectl rollout restart deployment/payment-svc",
                "expected_output": "deployment.apps/payment-svc restarted",
                "interpretation": "Deployment will be restarted",
                "why": "Restart clears transient state",
                "risk": "Brief downtime during restart",
                "related_findings": ["crashloop-payment-svc"],
            },
        ]

    return {
        "executive_summary": {
            "overall_status": "WARNING",
            "scope": "test-cluster/default",
            "summary_text": "Cluster has issues requiring attention",
        },
        "findings": findings,
        "recommendations": recommendations,
    }


# ══════════════════════════════════════════════════════════════
# REMEDIATION DISABLED
# ══════════════════════════════════════════════════════════════


class TestRemediateDisabled:
    """Tests when remediation is disabled in config."""

    def test_disabled_shows_message(self, cli_app: Any, runner: Any) -> None:
        """Exits with code 1 and shows 'disabled' panel."""
        settings = _make_settings(enabled=False)
        with patch("vaig.cli.commands.remediate._get_settings", return_value=settings):
            res = runner.invoke(cli_app, ["--list"])
        assert res.exit_code == 1
        output = _strip_ansi(res.output)
        assert "disabled" in output.lower()


# ══════════════════════════════════════════════════════════════
# NO ARGUMENTS
# ══════════════════════════════════════════════════════════════


class TestRemediateNoArgs:
    """Tests when no --list or --finding is specified."""

    def test_no_args_shows_error(self, cli_app: Any, runner: Any) -> None:
        """Exits with code 1 and instructs user."""
        settings = _make_settings()
        with patch("vaig.cli.commands.remediate._get_settings", return_value=settings):
            res = runner.invoke(cli_app, [])
        assert res.exit_code == 1
        output = _strip_ansi(res.output)
        assert "--list" in output or "--finding" in output


# ══════════════════════════════════════════════════════════════
# NO REPORT
# ══════════════════════════════════════════════════════════════


class TestRemediateNoReport:
    """Tests when no health report is available."""

    def test_no_report_shows_panel(self, cli_app: Any, runner: Any) -> None:
        """Exits with code 1 and suggests running a scan."""
        settings = _make_settings()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=None),
        ):
            res = runner.invoke(cli_app, ["--list"])
        assert res.exit_code == 1
        output = _strip_ansi(res.output)
        assert "no" in output.lower() and "report" in output.lower()


# ══════════════════════════════════════════════════════════════
# --list
# ══════════════════════════════════════════════════════════════


class TestRemediateList:
    """Tests for --list flag showing recommended actions."""

    def test_list_shows_table(self, cli_app: Any, runner: Any) -> None:
        """--list renders a table with recommendation details."""
        settings = _make_settings()
        report = _make_report_dict()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--list"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        # Rich table may truncate long text — check for key fragments
        assert "Restart" in output
        assert "crashloop-payment-svc" in output
        assert "SAFE" in output

    def test_list_no_recommendations(self, cli_app: Any, runner: Any) -> None:
        """--list with empty recommendations shows a green message."""
        settings = _make_settings()
        report = _make_report_dict(recommendations=[], findings=[])
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--list"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "no recommended" in output.lower()


# ══════════════════════════════════════════════════════════════
# --finding with SAFE tier + --approve
# ══════════════════════════════════════════════════════════════


class TestRemediateSafe:
    """Tests for SAFE-tier command remediation."""

    def test_safe_without_approve_prompts(self, cli_app: Any, runner: Any) -> None:
        """SAFE command without --approve shows prompt to re-run."""
        settings = _make_settings()
        report = _make_report_dict()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--finding", "crashloop-payment-svc"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "--approve" in output

    def test_safe_with_approve_executes(self, cli_app: Any, runner: Any) -> None:
        """SAFE command with --approve calls the executor."""
        settings = _make_settings()
        report = _make_report_dict()

        mock_result = MagicMock()
        mock_result.error = False
        mock_result.output = "deployment.apps/payment-svc restarted"

        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
            patch("vaig.cli.commands.remediate._execute_remediation") as mock_exec,
        ):
            res = runner.invoke(cli_app, ["--finding", "crashloop-payment-svc", "--approve"])
        assert res.exit_code == 0
        mock_exec.assert_called_once()


# ══════════════════════════════════════════════════════════════
# --finding with BLOCKED tier
# ══════════════════════════════════════════════════════════════


class TestRemediateBlocked:
    """Tests for BLOCKED-tier commands."""

    def test_blocked_command_shows_reason(self, cli_app: Any, runner: Any) -> None:
        """BLOCKED commands are never executed and show a reason panel."""
        settings = _make_settings()
        report = _make_report_dict(
            recommendations=[
                {
                    "priority": 1,
                    "title": "Delete failing pods",
                    "description": "Remove pods to trigger recreation",
                    "urgency": "high",
                    "effort": "low",
                    "command": "kubectl delete pods --all",
                    "expected_output": "pods deleted",
                    "interpretation": "All pods removed",
                    "why": "Force recreation",
                    "risk": "Service outage",
                    "related_findings": ["crashloop-payment-svc"],
                },
            ],
        )
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--finding", "crashloop-payment-svc"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "BLOCKED" in output.upper()


# ══════════════════════════════════════════════════════════════
# --finding with REVIEW tier
# ══════════════════════════════════════════════════════════════


class TestRemediateReview:
    """Tests for REVIEW-tier commands."""

    def test_review_without_execute_prompts(self, cli_app: Any, runner: Any) -> None:
        """REVIEW command without --execute shows prompt to re-run."""
        settings = _make_settings()
        report = _make_report_dict(
            recommendations=[
                {
                    "priority": 1,
                    "title": "Apply resource limits",
                    "description": "Set CPU/memory limits for stability",
                    "urgency": "medium",
                    "effort": "low",
                    "command": "kubectl apply -f limits.yaml",
                    "expected_output": "resource limits applied",
                    "interpretation": "Limits applied",
                    "why": "Prevents OOM kills",
                    "risk": "May throttle CPU",
                    "related_findings": ["crashloop-payment-svc"],
                },
            ],
        )
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--finding", "crashloop-payment-svc"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "--execute" in output

    def test_review_with_execute_calls_executor(self, cli_app: Any, runner: Any) -> None:
        """REVIEW command with --execute calls the executor."""
        settings = _make_settings()
        report = _make_report_dict(
            recommendations=[
                {
                    "priority": 1,
                    "title": "Apply resource limits",
                    "description": "Set CPU/memory limits for stability",
                    "urgency": "medium",
                    "effort": "low",
                    "command": "kubectl apply -f limits.yaml",
                    "expected_output": "resource limits applied",
                    "interpretation": "Limits applied",
                    "why": "Prevents OOM kills",
                    "risk": "May throttle CPU",
                    "related_findings": ["crashloop-payment-svc"],
                },
            ],
        )
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
            patch("vaig.cli.commands.remediate._execute_remediation") as mock_exec,
        ):
            res = runner.invoke(cli_app, ["--finding", "crashloop-payment-svc", "--execute"])
        assert res.exit_code == 0
        mock_exec.assert_called_once()


# ══════════════════════════════════════════════════════════════
# --dry-run
# ══════════════════════════════════════════════════════════════


class TestRemediateDryRun:
    """Tests for --dry-run flag."""

    def test_dry_run_shows_plan(self, cli_app: Any, runner: Any) -> None:
        """--dry-run shows the execution plan without running the command."""
        settings = _make_settings()
        report = _make_report_dict()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(
                cli_app,
                ["--finding", "crashloop-payment-svc", "--dry-run"],
            )
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "DRY RUN" in output.upper()
        assert "kubectl rollout restart" in output


# ══════════════════════════════════════════════════════════════
# FINDING NOT FOUND
# ══════════════════════════════════════════════════════════════


class TestFindingNotFound:
    """Tests when the specified finding ID doesn't match."""

    def test_unknown_finding_shows_error(self, cli_app: Any, runner: Any) -> None:
        """Unknown finding ID exits with code 1 and suggests --list."""
        settings = _make_settings()
        report = _make_report_dict()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--finding", "nonexistent-finding-id"])
        assert res.exit_code == 1
        output = _strip_ansi(res.output)
        assert "--list" in output


# ══════════════════════════════════════════════════════════════
# FINDING BY INDEX
# ══════════════════════════════════════════════════════════════


class TestFindingByIndex:
    """Tests finding lookup by numeric index."""

    def test_finding_by_index(self, cli_app: Any, runner: Any) -> None:
        """Passing a number as --finding matches by 1-based index."""
        settings = _make_settings()
        report = _make_report_dict()
        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
        ):
            res = runner.invoke(cli_app, ["--finding", "1", "--dry-run"])
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "DRY RUN" in output.upper()


# ══════════════════════════════════════════════════════════════
# EXECUTION — success and failure
# ══════════════════════════════════════════════════════════════


class TestRemediateExecution:
    """Tests for actual command execution paths."""

    def test_execution_success(self, cli_app: Any, runner: Any) -> None:
        """Successful execution displays a green success panel."""
        settings = _make_settings()
        report = _make_report_dict()

        mock_result = MagicMock()
        mock_result.error = False
        mock_result.output = "deployment.apps/payment-svc restarted"

        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
            patch("vaig.core.remediation.RemediationExecutor") as mock_exec_cls,
            patch("vaig.core.event_bus.EventBus", return_value=MagicMock()),
            patch("vaig.core.gke.build_gke_config", return_value=MagicMock()),
            patch("asyncio.run", return_value=mock_result),
        ):
            res = runner.invoke(
                cli_app,
                ["--finding", "crashloop-payment-svc", "--approve"],
            )
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "success" in output.lower()

    def test_execution_failure(self, cli_app: Any, runner: Any) -> None:
        """Failed execution displays a red error panel."""
        settings = _make_settings()
        report = _make_report_dict()

        mock_result = MagicMock()
        mock_result.error = True
        mock_result.output = "error: deployment not found"

        with (
            patch("vaig.cli.commands.remediate._get_settings", return_value=settings),
            patch("vaig.cli.commands.remediate._load_last_report", return_value=report),
            patch("vaig.core.remediation.RemediationExecutor") as mock_exec_cls,
            patch("vaig.core.event_bus.EventBus", return_value=MagicMock()),
            patch("vaig.core.gke.build_gke_config", return_value=MagicMock()),
            patch("asyncio.run", return_value=mock_result),
        ):
            res = runner.invoke(
                cli_app,
                ["--finding", "crashloop-payment-svc", "--approve"],
            )
        assert res.exit_code == 0
        output = _strip_ansi(res.output)
        assert "error" in output.lower()

"""Tests for ``vaig check`` CLI command — JSON output, exit codes, caching."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.cli.check_schema import CheckOutput
from vaig.cli.commands.check import _cache_key

runner = CliRunner()

# ── Fake objects for mocking ─────────────────────────────────


def _fake_orchestrator_result(
    overall_status: str = "HEALTHY",
    critical_count: int = 0,
    warning_count: int = 0,
    issues_found: int = 0,
    services_checked: int = 5,
    summary_text: str = "All services healthy",
    scope: str = "Namespace: default",
) -> MagicMock:
    """Build a mock OrchestratorResult with a structured HealthReport."""
    es = MagicMock()
    es.overall_status.value = overall_status
    es.critical_count = critical_count
    es.warning_count = warning_count
    es.issues_found = issues_found
    es.services_checked = services_checked
    es.summary_text = summary_text
    es.scope = scope

    report = MagicMock()
    report.executive_summary = es

    result = MagicMock()
    result.structured_report = report
    result.success = True
    return result


# Common patch targets — patch where imported, not where defined
_P_HEADLESS = "vaig.core.headless.execute_skill_headless"
_P_SETTINGS = "vaig.cli._helpers._get_settings"
_P_GKE = "vaig.cli.commands.check._build_gke_config"


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.gcp.project_id = "test-project"
    s.gke.cluster_name = "test-cluster"
    s.gke.default_namespace = "default"
    s.gke.location = "us-central1"
    s.gke.kubeconfig_path = ""
    s.gke.context = ""
    s.gke.project_id = ""
    s.models.default = "gemini-2.0-flash"
    s.logging.file_enabled = False
    s.logging.file_path = ""
    s.logging.file_level = "WARNING"
    s.logging.file_max_bytes = 0
    s.logging.file_backup_count = 0
    return s


def _mock_gke_config() -> MagicMock:
    cfg = MagicMock()
    cfg.cluster_name = "test-cluster"
    cfg.default_namespace = "default"
    cfg.location = "us-central1"
    cfg.project_id = "test-project"
    return cfg


# ── JSON output tests ────────────────────────────────────────


class TestCheckJsonOutput:
    """Verify stdout is valid JSON matching CheckOutput schema."""

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_stdout_is_valid_json(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result()
        result = runner.invoke(app, ["check"])

        parsed = json.loads(result.stdout)
        assert isinstance(parsed, dict)

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_output_matches_check_output_schema(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result()
        result = runner.invoke(app, ["check"])

        parsed = json.loads(result.stdout)
        # Must parse into a valid CheckOutput
        output = CheckOutput(**parsed)
        assert output.status == "HEALTHY"

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_all_values_are_strings(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result(critical_count=3)
        result = runner.invoke(app, ["check"])

        parsed = json.loads(result.stdout)
        for key, value in parsed.items():
            assert isinstance(value, str), f"Field '{key}' is {type(value).__name__}, not str"

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_no_ansi_in_stdout(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        """Stdout must contain zero ANSI escape sequences."""
        mock_headless.return_value = _fake_orchestrator_result()
        result = runner.invoke(app, ["check"])

        assert "\x1b[" not in result.stdout
        assert "\033[" not in result.stdout


# ── Exit code tests ──────────────────────────────────────────


class TestCheckExitCodes:
    """Verify exit codes match health status."""

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_healthy_exit_0(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result(overall_status="HEALTHY")
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_critical_exit_1(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result(
            overall_status="CRITICAL",
            critical_count=2,
        )
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_degraded_exit_1(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.return_value = _fake_orchestrator_result(overall_status="DEGRADED")
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_error_exit_2(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        mock_headless.side_effect = RuntimeError("Connection failed")
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 2
        # Even on error, stdout must be valid JSON
        parsed = json.loads(result.stdout)
        assert parsed["status"] == "ERROR"

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_critical_json_still_written(
        self, mock_settings: MagicMock, mock_gke: MagicMock, mock_headless: MagicMock
    ) -> None:
        """JSON must be written to stdout BEFORE exit, even on non-zero."""
        mock_headless.return_value = _fake_orchestrator_result(overall_status="CRITICAL")
        result = runner.invoke(app, ["check"])
        parsed = json.loads(result.stdout)
        assert parsed["status"] == "CRITICAL"


# ── Cache tests ──────────────────────────────────────────────


class TestCheckCache:
    """Verify --cached flag behaviour."""

    def test_cache_hit_returns_cached_result(self, tmp_path: Path) -> None:
        """When cache is fresh, --cached returns it without running pipeline."""
        key = _cache_key("default", "test-cluster", "test-project", "us-central1")
        output = CheckOutput.from_error("HEALTHY", "All good")
        output.status = "HEALTHY"

        # Write cache to temp dir
        cache_dir = tmp_path / "check"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / f"{key}.json"
        cache_file.write_text(output.model_dump_json())

        with (
            patch("vaig.cli.commands.check._CACHE_DIR", cache_dir),
            patch(_P_GKE, return_value=_mock_gke_config()),
            patch(_P_SETTINGS, return_value=_mock_settings()),
        ):
            result = runner.invoke(app, [
                "check", "--cached",
                "--namespace", "default",
                "--cluster", "test-cluster",
                "--project", "test-project",
            ])
            # Should succeed without calling the pipeline
            parsed = json.loads(result.stdout)
            assert parsed["cached"] == "true"

    @patch(_P_HEADLESS)
    @patch(_P_GKE, return_value=_mock_gke_config())
    @patch(_P_SETTINGS, return_value=_mock_settings())
    def test_cache_miss_runs_pipeline(
        self,
        mock_settings: MagicMock,
        mock_gke: MagicMock,
        mock_headless: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When no cache exists, --cached runs a fresh check."""
        mock_headless.return_value = _fake_orchestrator_result()
        cache_dir = tmp_path / "check"
        cache_dir.mkdir(parents=True)

        with patch("vaig.cli.commands.check._CACHE_DIR", cache_dir):
            result = runner.invoke(app, ["check", "--cached"])

        assert result.exit_code == 0
        mock_headless.assert_called_once()

    def test_stale_cache_runs_pipeline(self, tmp_path: Path) -> None:
        """When cache is older than TTL, --cached runs a fresh check."""
        key = _cache_key("default", "test-cluster", "test-project", "us-central1")
        output = CheckOutput.from_error("HEALTHY", "old result")
        output.status = "HEALTHY"

        cache_dir = tmp_path / "check"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / f"{key}.json"
        cache_file.write_text(output.model_dump_json())
        # Set mtime to 10 minutes ago
        old_time = time.time() - 600
        os.utime(cache_file, (old_time, old_time))

        with (
            patch("vaig.cli.commands.check._CACHE_DIR", cache_dir),
            patch(_P_HEADLESS) as mock_headless,
            patch(_P_GKE, return_value=_mock_gke_config()),
            patch(_P_SETTINGS, return_value=_mock_settings()),
        ):
            mock_headless.return_value = _fake_orchestrator_result()
            result = runner.invoke(app, ["check", "--cached", "--cache-ttl", "300"])

        mock_headless.assert_called_once()


# ── Cache key tests ──────────────────────────────────────────


class TestCacheKey:
    """Verify cache key generation."""

    def test_deterministic(self) -> None:
        k1 = _cache_key("ns", "cluster", "project", "us-central1")
        k2 = _cache_key("ns", "cluster", "project", "us-central1")
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        k1 = _cache_key("ns1", "cluster", "project", "us-central1")
        k2 = _cache_key("ns2", "cluster", "project", "us-central1")
        assert k1 != k2

    def test_none_handling(self) -> None:
        k1 = _cache_key(None, None, None, None)
        assert isinstance(k1, str)
        assert len(k1) == 64  # SHA-256 hex digest

    def test_location_affects_key(self) -> None:
        k1 = _cache_key("ns", "cluster", "project", "us-central1")
        k2 = _cache_key("ns", "cluster", "project", "europe-west1")
        assert k1 != k2

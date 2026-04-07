"""CLI integration tests for cross-cluster comparison."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
from typer.testing import CliRunner

from vaig.cli.commands.compare import compare_app
from vaig.core.compare import CompareMetadata, CompareReport, DeploymentSnapshot, FieldDiff
from vaig.core.config import FleetCluster, FleetConfig

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip ANSI escape codes."""
    return click.unstyle(output)


def _make_snapshot(cluster: str, image_tag: str = "v2.1.0") -> DeploymentSnapshot:
    """Helper to build a minimal DeploymentSnapshot."""
    from datetime import UTC, datetime

    return DeploymentSnapshot(
        cluster_name=cluster,
        namespace="default",
        deployment_name="api",
        image_tag=image_tag,
        replicas_desired=3,
        replicas_ready=3,
        rollout_generation=5,
        hpa_min=2,
        hpa_max=10,
        collected_at=datetime.now(tz=UTC),
    )


class TestCompareCLI:
    """REQ-CMP-05: CLI command tests."""

    def test_compare_help(self) -> None:
        """--help shows all expected flags."""
        result = runner.invoke(compare_app, ["--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        text = _plain(result.output)
        assert "--clusters" in text
        assert "--namespace" in text
        assert "--deployment" in text
        assert "--export" in text

    def test_no_fleet_config_error(self) -> None:
        """No fleet clusters → actionable error."""
        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(clusters=[])

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            result = runner.invoke(
                compare_app,
                ["--clusters", "a,b", "--namespace", "default", "--deployment", "api"],
            )

        assert result.exit_code == 1
        assert "No fleet clusters configured" in result.output

    def test_cluster_not_found(self) -> None:
        """SC-07: Cluster name not in config → error with available names."""
        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[FleetCluster(name="prod-us", cluster_name="gke-us")],
        )

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            result = runner.invoke(
                compare_app,
                ["--clusters", "prod-us,nonexistent", "-n", "default", "-d", "api"],
            )

        assert result.exit_code == 1
        assert "nonexistent" in result.output
        assert "prod-us" in result.output  # shows available

    def test_fewer_than_two_clusters(self) -> None:
        """At least 2 clusters required for comparison."""
        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[FleetCluster(name="prod-us", cluster_name="gke-us")],
        )

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            result = runner.invoke(
                compare_app,
                ["--clusters", "prod-us", "-n", "default", "-d", "api"],
            )

        assert result.exit_code == 1
        assert "At least 2 clusters" in result.output

    def test_full_run_with_mock_runner(self) -> None:
        """Full run with mocked CompareRunner → displays table."""
        mock_report = CompareReport(
            snapshots={
                "prod-us": _make_snapshot("gke-us"),
                "prod-eu": _make_snapshot("gke-eu", image_tag="v2.0.3"),
            },
            errors={},
            diffs=[
                FieldDiff(
                    field="image_tag",
                    values={"prod-us": "v2.1.0", "prod-eu": "v2.0.3"},
                    severity="critical",
                ),
            ],
            metadata=CompareMetadata(
                clusters_requested=["prod-us", "prod-eu"],
                namespace="default",
                deployment="api",
            ),
        )

        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[
                FleetCluster(name="prod-us", cluster_name="gke-us"),
                FleetCluster(name="prod-eu", cluster_name="gke-eu"),
            ],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_parallel.return_value = mock_report

        with patch("vaig.core.config.get_settings", return_value=mock_settings), \
             patch("vaig.core.compare.CompareRunner", return_value=mock_runner_instance):
            result = runner.invoke(
                compare_app,
                ["--clusters", "prod-us,prod-eu", "-n", "default", "-d", "api"],
            )

        assert result.exit_code == 0
        output = _plain(result.output)
        assert "Cross-Cluster Compare" in result.output or "Cross-Cluster Comparison" in output
        assert "divergence" in output.lower() or "image_tag" in output

    def test_export_json(self) -> None:
        """SC-06: --export json produces JSON output."""
        mock_report = CompareReport(
            snapshots={
                "prod-us": _make_snapshot("gke-us"),
                "prod-eu": _make_snapshot("gke-eu"),
            },
            errors={},
            diffs=[],
            metadata=CompareMetadata(
                clusters_requested=["prod-us", "prod-eu"],
                namespace="default",
                deployment="api",
            ),
        )

        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[
                FleetCluster(name="prod-us", cluster_name="gke-us"),
                FleetCluster(name="prod-eu", cluster_name="gke-eu"),
            ],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_parallel.return_value = mock_report

        with patch("vaig.core.config.get_settings", return_value=mock_settings), \
             patch("vaig.core.compare.CompareRunner", return_value=mock_runner_instance):
            result = runner.invoke(
                compare_app,
                [
                    "--clusters", "prod-us,prod-eu",
                    "-n", "default",
                    "-d", "api",
                    "--export", "json",
                ],
            )

        assert result.exit_code == 0
        assert '"snapshots"' in result.output
        assert '"diffs"' in result.output
        assert '"metadata"' in result.output

    def test_all_clusters_fail(self) -> None:
        """All clusters failing → exit code 1."""
        mock_report = CompareReport(
            snapshots={},
            errors={"prod-us": "timeout", "prod-eu": "auth failed"},
            diffs=[],
            metadata=CompareMetadata(
                clusters_requested=["prod-us", "prod-eu"],
                namespace="default",
                deployment="api",
            ),
        )

        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[
                FleetCluster(name="prod-us", cluster_name="gke-us"),
                FleetCluster(name="prod-eu", cluster_name="gke-eu"),
            ],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run_parallel.return_value = mock_report

        with patch("vaig.core.config.get_settings", return_value=mock_settings), \
             patch("vaig.core.compare.CompareRunner", return_value=mock_runner_instance):
            result = runner.invoke(
                compare_app,
                ["--clusters", "prod-us,prod-eu", "-n", "default", "-d", "api"],
            )

        assert result.exit_code == 1
        assert "All clusters failed" in result.output or "No snapshots" in result.output

"""CLI integration tests for fleet scanning — T-19."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
from typer.testing import CliRunner

from vaig.cli.commands.fleet import fleet_app
from vaig.core.config import FleetCluster, FleetConfig

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip ANSI escape codes so assertions work regardless of terminal width."""
    return click.unstyle(output)


class TestFleetCLI:
    """REQ-FLEET-05, REQ-FLEET-06, SC-05: CLI command tests."""

    def test_fleet_discover_help(self) -> None:
        """--help shows all expected flags."""
        result = runner.invoke(fleet_app, ["discover", "--help"])
        assert result.exit_code == 0
        text = _plain(result.output)
        assert "--parallel" in text
        assert "--max-workers" in text
        assert "--budget" in text
        assert "--detailed" in text
        assert "--export" in text
        assert "--namespace" in text
        assert "--all-namespaces" in text

    def test_no_fleet_config_error(self) -> None:
        """No fleet config → actionable error."""
        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(clusters=[])

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            result = runner.invoke(fleet_app, [])

        assert result.exit_code == 1
        assert "No fleet clusters configured" in result.output

    def test_fleet_discover_with_mock_runner(self) -> None:
        """Full run with mocked FleetRunner."""
        from vaig.core.fleet import ClusterResult, FleetReport

        mock_report = FleetReport(
            clusters=[
                ClusterResult(
                    cluster_name="gke-0",
                    display_name="prod-us",
                    status="success",
                    cost_usd=0.5,
                    duration_s=2.0,
                ),
                ClusterResult(
                    cluster_name="gke-1",
                    display_name="prod-eu",
                    status="success",
                    cost_usd=0.4,
                    duration_s=1.5,
                ),
            ],
            correlations=[],
            total_duration_s=3.5,
            total_cost_usd=0.9,
        )

        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[
                FleetCluster(name="prod-us", cluster_name="gke-0"),
                FleetCluster(name="prod-eu", cluster_name="gke-1"),
            ],
        )

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_report

        with patch("vaig.core.config.get_settings", return_value=mock_settings), \
             patch("vaig.core.fleet.FleetRunner", return_value=mock_runner):
            result = runner.invoke(fleet_app, [])

        # Should succeed (not all failed)
        assert result.exit_code == 0
        assert "Fleet Scan Summary" in result.output

    def test_fleet_discover_export_json(self) -> None:
        """--export json produces JSON output."""
        from vaig.core.fleet import ClusterResult, FleetReport

        mock_report = FleetReport(
            clusters=[
                ClusterResult(cluster_name="gke-0", display_name="prod", status="success", cost_usd=0.5),
            ],
            total_cost_usd=0.5,
            total_duration_s=1.0,
        )

        mock_settings = MagicMock()
        mock_settings.fleet = FleetConfig(
            clusters=[FleetCluster(name="prod", cluster_name="gke-0")],
        )

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_report

        with patch("vaig.core.config.get_settings", return_value=mock_settings), \
             patch("vaig.core.fleet.FleetRunner", return_value=mock_runner):
            result = runner.invoke(fleet_app, ["--export", "json"])

        assert result.exit_code == 0
        # JSON export should contain the clusters key
        assert '"clusters"' in result.output
        assert '"metadata"' in result.output

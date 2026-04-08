"""Tests for CLI incident commands — export, list, error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from vaig.cli.commands.incident_cmd import incident_app

runner = CliRunner()


# ── Export command tests ─────────────────────────────────────


class TestExportCommand:
    """Tests for `vaig incident export`."""

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    @patch("vaig.cli.commands.incident_cmd.ReportStore")
    @patch("vaig.cli.commands.incident_cmd.FindingExporter")
    @patch("vaig.cli.commands.incident_cmd.JiraClient")
    def test_export_jira_success(
        self,
        mock_jira_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        from vaig.integrations.finding_exporter import ExportResult

        mock_settings.return_value.jira.enabled = True
        mock_settings.return_value.jira.base_url = "https://myorg.atlassian.net"
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = ExportResult(
            target="jira",
            success=True,
            url="https://myorg.atlassian.net/browse/OPS-42",
            key="OPS-42",
        )

        result = runner.invoke(
            incident_app,
            ["export", "--to", "jira", "--finding", "crashloop-payment"],
        )

        assert result.exit_code == 0
        assert "OPS-42" in result.output

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    def test_export_jira_not_configured(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.jira.enabled = False

        result = runner.invoke(
            incident_app,
            ["export", "--to", "jira", "--finding", "any-slug"],
        )

        assert result.exit_code != 0

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    def test_export_pagerduty_not_configured(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.pagerduty.enabled = False

        result = runner.invoke(
            incident_app,
            ["export", "--to", "pagerduty", "--finding", "any-slug"],
        )

        assert result.exit_code != 0

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    def test_export_unknown_target(self, mock_settings: MagicMock) -> None:
        result = runner.invoke(
            incident_app,
            ["export", "--to", "unknown", "--finding", "any-slug"],
        )

        assert result.exit_code != 0

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    @patch("vaig.cli.commands.incident_cmd.ReportStore")
    @patch("vaig.cli.commands.incident_cmd.FindingExporter")
    @patch("vaig.cli.commands.incident_cmd.JiraClient")
    def test_export_finding_not_found(
        self,
        mock_jira_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        from vaig.integrations.finding_exporter import ExportResult

        mock_settings.return_value.jira.enabled = True
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = ExportResult(
            target="jira",
            success=False,
            error="Finding 'nonexistent' not found.",
        )

        result = runner.invoke(
            incident_app,
            ["export", "--to", "jira", "--finding", "nonexistent"],
        )

        assert result.exit_code != 0

    @patch("vaig.cli.commands.incident_cmd.get_settings")
    @patch("vaig.cli.commands.incident_cmd.ReportStore")
    @patch("vaig.cli.commands.incident_cmd.FindingExporter")
    @patch("vaig.cli.commands.incident_cmd.JiraClient")
    def test_export_already_existed_shows_warning(
        self,
        mock_jira_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        from vaig.integrations.finding_exporter import ExportResult

        mock_settings.return_value.jira.enabled = True
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = ExportResult(
            target="jira",
            success=True,
            url="https://myorg.atlassian.net/browse/OPS-42",
            key="OPS-42",
            already_existed=True,
        )

        result = runner.invoke(
            incident_app,
            ["export", "--to", "jira", "--finding", "crashloop-payment"],
        )

        assert result.exit_code == 0
        assert "already exported" in result.output.lower()


# ── List command tests ──────────────────────────────────────


class TestListCommand:
    """Tests for `vaig incident list`."""

    @patch("vaig.cli.commands.incident_cmd.ReportStore")
    @patch("vaig.cli.commands.incident_cmd.FindingExporter")
    def test_list_shows_findings(
        self, mock_exporter_cls: MagicMock, mock_store_cls: MagicMock
    ) -> None:
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.list_findings.return_value = [
            {
                "id": "crashloop-payment",
                "title": "CrashLoop in payment-svc",
                "severity": "HIGH",
                "service": "payment-svc",
                "timestamp": "2026-04-08T12:00:00",
                "run_id": "run-001",
            }
        ]

        result = runner.invoke(incident_app, ["list"])

        assert result.exit_code == 0
        assert "crashloop-payment" in result.output

    @patch("vaig.cli.commands.incident_cmd.ReportStore")
    @patch("vaig.cli.commands.incident_cmd.FindingExporter")
    def test_list_no_findings(
        self, mock_exporter_cls: MagicMock, mock_store_cls: MagicMock
    ) -> None:
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.list_findings.return_value = []

        result = runner.invoke(incident_app, ["list"])

        assert result.exit_code == 0
        assert "no findings" in result.output.lower()

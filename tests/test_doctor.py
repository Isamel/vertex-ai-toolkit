"""Tests for the doctor command — environment healthcheck.

Validates:
- Each check function returns the correct CheckResult for pass/warn/fail scenarios
- Output formatting (icons, alignment)
- Exit code 1 when critical checks fail, 0 when they pass
- Failures in one check don't prevent other checks from running
- ALL external dependencies are mocked (no real GCP/GKE/network calls)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.cli.commands.doctor import (
    CheckResult,
    check_argocd,
    check_cloud_logging,
    check_cloud_monitoring,
    check_datadog,
    check_gcp_auth,
    check_gke_connectivity,
    check_helm,
    check_mcp,
    check_optional_deps,
    check_vertex_ai,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_settings(**overrides: Any) -> SimpleNamespace:
    """Build a fake settings object that mimics the real Settings model."""
    defaults = {
        "gcp": SimpleNamespace(project_id="test-project", location="us-central1"),
        "auth": SimpleNamespace(),
        "models": SimpleNamespace(default="gemini-2.0-flash"),
        "gke": SimpleNamespace(
            cluster_name="test-cluster",
            kubeconfig_path=None,
            context=None,
            project_id=None,
            location=None,
        ),
        "helm": SimpleNamespace(enabled=True),
        "argocd": SimpleNamespace(enabled=False, server="", token=""),
        "datadog": SimpleNamespace(
            enabled=False,
            api_key="",
            app_key="",
            site="datadoghq.com",
            ssl_verify=True,
        ),
        "mcp": SimpleNamespace(enabled=False, servers=[]),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── CheckResult model tests ──────────────────────────────────


class TestCheckResult:
    """CheckResult dataclass and icon property."""

    def test_pass_icon(self) -> None:
        r = CheckResult(name="test", status="pass", message="ok")
        assert "green" in r.icon
        assert "\u2713" in r.icon

    def test_warn_icon(self) -> None:
        r = CheckResult(name="test", status="warn", message="warning")
        assert "yellow" in r.icon
        assert "\u26a0" in r.icon

    def test_fail_icon(self) -> None:
        r = CheckResult(name="test", status="fail", message="error")
        assert "red" in r.icon
        assert "\u2717" in r.icon

    def test_unknown_status_shows_fail_icon(self) -> None:
        r = CheckResult(name="test", status="unknown", message="?")
        assert "red" in r.icon


# ── GCP Auth check tests ─────────────────────────────────────


class TestCheckGCPAuth:
    """GCP Application Default Credentials check."""

    @patch("google.auth.default")
    def test_pass_with_adc(self, mock_default: MagicMock) -> None:
        mock_default.return_value = (MagicMock(), "detected-project")
        settings = _make_settings()
        result = check_gcp_auth(settings)
        assert result.status == "pass"
        assert result.name == "GCP Authentication"
        assert "test-project" in result.message

    @patch("google.auth.default")
    def test_pass_uses_detected_project_when_no_override(self, mock_default: MagicMock) -> None:
        mock_default.return_value = (MagicMock(), "detected-project")
        settings = _make_settings(gcp=SimpleNamespace(project_id="", location="us-central1"))
        result = check_gcp_auth(settings)
        assert result.status == "pass"
        assert "detected-project" in result.message

    @patch("google.auth.default")
    def test_pass_unknown_project_fallback(self, mock_default: MagicMock) -> None:
        mock_default.return_value = (MagicMock(), None)
        settings = _make_settings(gcp=SimpleNamespace(project_id="", location="us-central1"))
        result = check_gcp_auth(settings)
        assert result.status == "pass"
        assert "unknown" in result.message

    @patch("google.auth.default", side_effect=Exception("Could not find default credentials"))
    def test_fail_when_no_adc(self, mock_default: MagicMock) -> None:
        settings = _make_settings()
        result = check_gcp_auth(settings)
        assert result.status == "fail"
        assert "gcloud auth application-default login" in result.message


# ── Vertex AI check tests ────────────────────────────────────


class TestCheckVertexAI:
    """Vertex AI API accessibility check."""

    @patch("google.genai.Client")
    @patch("vaig.core.auth.get_credentials")
    def test_pass_when_model_accessible(
        self, mock_get_creds: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        settings = _make_settings()
        result = check_vertex_ai(settings)
        assert result.status == "pass"
        assert "gemini-2.0-flash" in result.message
        mock_client.models.count_tokens.assert_called_once()

    @patch("google.genai.Client")
    @patch("vaig.core.auth.get_credentials")
    def test_fail_when_api_unreachable(
        self, mock_get_creds: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.models.count_tokens.side_effect = Exception("403 Forbidden")
        mock_client_cls.return_value = mock_client
        settings = _make_settings()
        result = check_vertex_ai(settings)
        assert result.status == "fail"
        assert "not reachable" in result.message

    @patch("vaig.core.auth.get_credentials", side_effect=Exception("credentials error"))
    def test_fail_when_credentials_fail(self, mock_get_creds: MagicMock) -> None:
        settings = _make_settings()
        result = check_vertex_ai(settings)
        assert result.status == "fail"


# ── GKE Connectivity check tests ─────────────────────────────


class TestCheckGKEConnectivity:
    """GKE / Kubernetes cluster connectivity check."""

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.client.VersionApi")
    def test_pass_with_cluster(
        self, mock_version_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        mock_version = MagicMock()
        mock_version.git_version = "v1.28.3"
        mock_version_api.return_value.get_code.return_value = mock_version
        settings = _make_settings()
        result = check_gke_connectivity(settings)
        assert result.status == "pass"
        assert "test-cluster" in result.message
        assert "v1.28.3" in result.message

    @patch("kubernetes.config.load_kube_config")
    @patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster"))
    @patch("kubernetes.client.VersionApi")
    def test_pass_fallback_to_kubeconfig(
        self,
        mock_version_api: MagicMock,
        mock_incluster: MagicMock,
        mock_kube_config: MagicMock,
    ) -> None:
        mock_version = MagicMock()
        mock_version.git_version = "v1.29.0"
        mock_version_api.return_value.get_code.return_value = mock_version
        # ConfigException is the specific exception that triggers the fallback
        from kubernetes.config import ConfigException

        mock_incluster.side_effect = ConfigException("not in cluster")
        settings = _make_settings()
        result = check_gke_connectivity(settings)
        assert result.status == "pass"

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.client.VersionApi")
    def test_fail_when_cluster_unreachable(
        self, mock_version_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        mock_version_api.return_value.get_code.side_effect = Exception("Unable to connect")
        settings = _make_settings()
        result = check_gke_connectivity(settings)
        assert result.status == "fail"
        assert "cannot connect" in result.message

    @patch("kubernetes.config.load_kube_config")
    @patch("kubernetes.client.VersionApi")
    def test_uses_kubeconfig_path_when_set(
        self, mock_version_api: MagicMock, mock_load: MagicMock
    ) -> None:
        mock_version = MagicMock()
        mock_version.git_version = "v1.28.0"
        mock_version_api.return_value.get_code.return_value = mock_version
        settings = _make_settings(
            gke=SimpleNamespace(
                cluster_name="custom",
                kubeconfig_path="/custom/kubeconfig",
                context="my-context",
                project_id=None,
                location=None,
            )
        )
        result = check_gke_connectivity(settings)
        assert result.status == "pass"
        mock_load.assert_called_once_with(
            config_file="/custom/kubeconfig",
            context="my-context",
        )


# ── Cloud Logging check tests ────────────────────────────────


class TestCheckCloudLogging:
    """Cloud Logging API check."""

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_pass_when_client_ready(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = (MagicMock(), None)
        settings = _make_settings()
        result = check_cloud_logging(settings)
        assert result.status == "pass"
        assert "client ready" in result.message

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_fail_when_client_returns_error(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = (None, "SDK not installed")
        settings = _make_settings()
        result = check_cloud_logging(settings)
        assert result.status == "fail"
        assert "SDK not installed" in result.message

    @patch("vaig.tools.gcloud_tools._get_logging_client", side_effect=Exception("boom"))
    def test_fail_when_exception(self, mock_get_client: MagicMock) -> None:
        settings = _make_settings()
        result = check_cloud_logging(settings)
        assert result.status == "fail"
        assert "check failed" in result.message


# ── Cloud Monitoring check tests ─────────────────────────────


class TestCheckCloudMonitoring:
    """Cloud Monitoring API check."""

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_pass_when_client_ready(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = (MagicMock(), None)
        settings = _make_settings()
        result = check_cloud_monitoring(settings)
        assert result.status == "pass"
        assert "client ready" in result.message

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_fail_when_client_returns_error(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = (None, "SDK not installed")
        settings = _make_settings()
        result = check_cloud_monitoring(settings)
        assert result.status == "fail"
        assert "SDK not installed" in result.message


# ── Helm check tests ─────────────────────────────────────────


class TestCheckHelm:
    """Helm integration check."""

    @patch("shutil.which", return_value="/usr/local/bin/helm")
    def test_pass_when_enabled_and_found(self, mock_which: MagicMock) -> None:
        settings = _make_settings(helm=SimpleNamespace(enabled=True))
        result = check_helm(settings)
        assert result.status == "pass"
        assert "/usr/local/bin/helm" in result.message

    @patch("shutil.which", return_value=None)
    def test_warn_when_enabled_but_not_found(self, mock_which: MagicMock) -> None:
        settings = _make_settings(helm=SimpleNamespace(enabled=True))
        result = check_helm(settings)
        assert result.status == "warn"
        assert "not found" in result.message

    def test_warn_when_disabled(self) -> None:
        settings = _make_settings(helm=SimpleNamespace(enabled=False))
        result = check_helm(settings)
        assert result.status == "warn"
        assert "disabled" in result.message


# ── ArgoCD check tests ───────────────────────────────────────


class TestCheckArgoCD:
    """ArgoCD integration check."""

    def test_warn_when_disabled(self) -> None:
        settings = _make_settings()
        result = check_argocd(settings)
        assert result.status == "warn"
        assert "disabled" in result.message

    def test_warn_when_enabled_no_server(self) -> None:
        settings = _make_settings(
            argocd=SimpleNamespace(enabled=True, server="", token="tok")
        )
        result = check_argocd(settings)
        assert result.status == "warn"
        assert "server not configured" in result.message

    def test_pass_when_fully_configured(self) -> None:
        settings = _make_settings(
            argocd=SimpleNamespace(
                enabled=True, server="argocd.example.com", token="tok"
            )
        )
        result = check_argocd(settings)
        assert result.status == "pass"
        assert "argocd.example.com" in result.message


# ── Datadog check tests ──────────────────────────────────────


class TestCheckDatadog:
    """Datadog API integration check."""

    def test_warn_when_disabled_no_keys(self) -> None:
        settings = _make_settings()
        result = check_datadog(settings)
        assert result.status == "warn"
        assert "disabled" in result.message

    def test_warn_when_disabled_with_keys(self) -> None:
        settings = _make_settings(
            datadog=SimpleNamespace(
                enabled=False,
                api_key="key1",
                app_key="key2",
                site="datadoghq.com",
                ssl_verify=True,
            )
        )
        result = check_datadog(settings)
        assert result.status == "warn"
        assert "keys present but disabled" in result.message

    def test_pass_when_enabled(self) -> None:
        settings = _make_settings(
            datadog=SimpleNamespace(
                enabled=True,
                api_key="key1",
                app_key="key2",
                site="datadoghq.com",
                ssl_verify=True,
            )
        )
        result = check_datadog(settings)
        assert result.status == "pass"
        assert "datadoghq.com" in result.message

    def test_pass_ssl_verify_false_note(self) -> None:
        settings = _make_settings(
            datadog=SimpleNamespace(
                enabled=True,
                api_key="key1",
                app_key="key2",
                site="datadoghq.com",
                ssl_verify=False,
            )
        )
        result = check_datadog(settings)
        assert result.status == "pass"
        assert "ssl_verify=False" in result.message

    def test_pass_custom_ca_note(self) -> None:
        settings = _make_settings(
            datadog=SimpleNamespace(
                enabled=True,
                api_key="key1",
                app_key="key2",
                site="datadoghq.com",
                ssl_verify="/path/to/ca.pem",
            )
        )
        result = check_datadog(settings)
        assert result.status == "pass"
        assert "custom CA" in result.message


# ── Optional deps check tests ────────────────────────────────


class TestCheckOptionalDeps:
    """Optional dependencies importability check."""

    def test_runs_without_crashing(self) -> None:
        """Smoke test — runs the real check, shouldn't crash."""
        result = check_optional_deps()
        assert result.status in {"pass", "warn"}
        assert result.name == "Optional deps"

    def test_reports_available_and_missing(self) -> None:
        """Result message always provides useful information."""
        result = check_optional_deps()
        # At minimum, the message should be non-empty
        assert result.message


# ── MCP check tests ──────────────────────────────────────────


class TestCheckMCP:
    """MCP server configuration check."""

    def test_warn_when_disabled(self) -> None:
        settings = _make_settings()
        result = check_mcp(settings)
        assert result.status == "warn"
        assert "disabled" in result.message

    def test_warn_when_enabled_no_servers(self) -> None:
        settings = _make_settings(mcp=SimpleNamespace(enabled=True, servers=[]))
        result = check_mcp(settings)
        assert result.status == "warn"
        assert "no servers" in result.message

    def test_pass_with_servers(self) -> None:
        server = SimpleNamespace(name="my-mcp-server")
        settings = _make_settings(
            mcp=SimpleNamespace(enabled=True, servers=[server])
        )
        result = check_mcp(settings)
        assert result.status == "pass"
        assert "my-mcp-server" in result.message
        assert "1 server" in result.message

    def test_pass_with_multiple_servers(self) -> None:
        servers = [
            SimpleNamespace(name="server-a"),
            SimpleNamespace(name="server-b"),
        ]
        settings = _make_settings(
            mcp=SimpleNamespace(enabled=True, servers=servers)
        )
        result = check_mcp(settings)
        assert result.status == "pass"
        assert "2 server(s)" in result.message
        assert "server-a" in result.message
        assert "server-b" in result.message


# ── Integration: command execution tests ─────────────────────


def _patch_all_checks():
    """Context manager that patches all check functions and _get_settings."""
    return (
        patch("vaig.cli.commands.doctor._get_settings"),
        patch("vaig.cli.commands.doctor.check_gcp_auth"),
        patch("vaig.cli.commands.doctor.check_vertex_ai"),
        patch("vaig.cli.commands.doctor.check_gke_connectivity"),
        patch("vaig.cli.commands.doctor.check_cloud_logging"),
        patch("vaig.cli.commands.doctor.check_cloud_monitoring"),
        patch("vaig.cli.commands.doctor.check_helm"),
        patch("vaig.cli.commands.doctor.check_argocd"),
        patch("vaig.cli.commands.doctor.check_datadog"),
        patch("vaig.cli.commands.doctor.check_optional_deps"),
        patch("vaig.cli.commands.doctor.check_mcp"),
    )


def _configure_mocks(
    mocks: dict[str, MagicMock],
    *,
    auth: str = "pass",
    vertex: str = "pass",
    gke: str = "pass",
    logging: str = "pass",
    monitoring: str = "pass",
    helm: str = "pass",
    argocd: str = "warn",
    datadog: str = "warn",
    deps: str = "pass",
    mcp: str = "warn",
) -> None:
    """Configure mock check functions to return given statuses."""
    mocks["settings"].return_value = _make_settings()
    mocks["auth"].return_value = CheckResult("GCP Authentication", auth, f"{auth}")
    mocks["vertex"].return_value = CheckResult("Vertex AI API", vertex, f"{vertex}")
    mocks["gke"].return_value = CheckResult("GKE Connectivity", gke, f"{gke}")
    mocks["logging"].return_value = CheckResult("Cloud Logging", logging, f"{logging}")
    mocks["monitoring"].return_value = CheckResult("Cloud Monitoring", monitoring, f"{monitoring}")
    mocks["helm"].return_value = CheckResult("Helm Integration", helm, f"{helm}")
    mocks["argocd"].return_value = CheckResult("ArgoCD Integration", argocd, f"{argocd}")
    mocks["datadog"].return_value = CheckResult("Datadog Integration", datadog, f"{datadog}")
    mocks["deps"].return_value = CheckResult("Optional deps", deps, f"{deps}")
    mocks["mcp"].return_value = CheckResult("MCP Servers", mcp, f"{mcp}")


@pytest.fixture()
def cli_app():
    """Create a fresh Typer app with doctor registered."""
    import typer

    from vaig.cli.commands.doctor import register

    test_app = typer.Typer()
    register(test_app)
    return test_app


@pytest.fixture()
def runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()


class TestDoctorCommand:
    """Integration tests via typer CliRunner."""

    def test_all_checks_pass_exit_0(self, cli_app: Any, runner: Any) -> None:
        """Doctor returns exit code 0 when critical checks pass."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            _configure_mocks(mocks)
            res = runner.invoke(cli_app, [])
            assert res.exit_code == 0

    def test_critical_gcp_auth_failure_exit_1(self, cli_app: Any, runner: Any) -> None:
        """Doctor returns exit code 1 when GCP Auth fails."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            _configure_mocks(mocks, auth="fail")
            res = runner.invoke(cli_app, [])
            assert res.exit_code == 1

    def test_critical_vertex_failure_exit_1(self, cli_app: Any, runner: Any) -> None:
        """Doctor returns exit code 1 when Vertex AI fails."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            _configure_mocks(mocks, vertex="fail")
            res = runner.invoke(cli_app, [])
            assert res.exit_code == 1

    def test_non_critical_failures_still_exit_0(self, cli_app: Any, runner: Any) -> None:
        """Doctor returns exit code 0 when only non-critical checks fail."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            _configure_mocks(
                mocks,
                gke="fail",
                logging="fail",
                monitoring="fail",
                helm="fail",
                argocd="fail",
                datadog="fail",
                mcp="fail",
            )
            res = runner.invoke(cli_app, [])
            assert res.exit_code == 0

    def test_output_contains_all_check_names(self, cli_app: Any, runner: Any) -> None:
        """Doctor output shows all check names."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            _configure_mocks(mocks)
            res = runner.invoke(cli_app, [])
            output = res.output
            for name in [
                "GCP Authentication",
                "Vertex AI API",
                "GKE Connectivity",
                "Cloud Logging",
                "Cloud Monitoring",
                "Helm Integration",
                "ArgoCD Integration",
                "Datadog Integration",
                "Optional deps",
                "MCP Servers",
            ]:
                assert name in output, f"Missing check name in output: {name}"

    def test_summary_counts(self, cli_app: Any, runner: Any) -> None:
        """Doctor output includes summary with pass/warn/fail counts."""
        patches = _patch_all_checks()
        with patches[0] as m_settings, patches[1] as m_auth, patches[2] as m_vertex, \
             patches[3] as m_gke, patches[4] as m_logging, patches[5] as m_monitoring, \
             patches[6] as m_helm, patches[7] as m_argocd, patches[8] as m_datadog, \
             patches[9] as m_deps, patches[10] as m_mcp:
            mocks = {
                "settings": m_settings, "auth": m_auth, "vertex": m_vertex,
                "gke": m_gke, "logging": m_logging, "monitoring": m_monitoring,
                "helm": m_helm, "argocd": m_argocd, "datadog": m_datadog,
                "deps": m_deps, "mcp": m_mcp,
            }
            # 7 pass, 2 warn, 1 fail
            _configure_mocks(
                mocks,
                argocd="warn",
                datadog="fail",
                mcp="warn",
            )
            res = runner.invoke(cli_app, [])
            output = res.output
            assert "7 passed" in output
            assert "2 warnings" in output
            assert "1 failed" in output

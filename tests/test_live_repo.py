"""Tests for GH-02 Live Mode Repo Correlation.

Covers:
- T-09: ConfigDriftFinding Pydantic model validation
- T-10: ``--repo`` / ``--repo-ref`` CLI options are forwarded correctly
- T-11: ``register_live_tools`` injects GitHub repo tools when enabled
- T-12: ``_planner.py`` contains commit_correlation and chart_regression hint types
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.gke import register_live_tools
from vaig.skills.service_health.schema import ConfigDriftFinding, Severity

# ── T-09: ConfigDriftFinding model ────────────────────────────────────────────


class TestConfigDriftFinding:
    """Tests for the ConfigDriftFinding Pydantic model (T-09)."""

    __test__ = True

    def test_minimal_valid_instance(self) -> None:
        """ConfigDriftFinding can be created with only required fields."""
        finding = ConfigDriftFinding(
            commit_sha="abc123def456",
            description="Suspected config drift from recent commit",
        )
        assert finding.finding_type == "config_drift"
        assert finding.commit_sha == "abc123def456"
        assert finding.severity == Severity.HIGH
        assert finding.changed_files == []
        assert finding.suspected_config_keys == []

    def test_full_instance(self) -> None:
        """ConfigDriftFinding stores all optional fields correctly."""
        finding = ConfigDriftFinding(
            commit_sha="deadbeef",
            changed_files=["config/prod.yaml", "helm/values.yaml"],
            suspected_config_keys=["MAX_CONNECTIONS", "DB_POOL_SIZE"],
            severity=Severity.CRITICAL,
            description="Connection pool increased beyond node limit",
        )
        assert finding.changed_files == ["config/prod.yaml", "helm/values.yaml"]
        assert finding.suspected_config_keys == ["MAX_CONNECTIONS", "DB_POOL_SIZE"]
        assert finding.severity == Severity.CRITICAL

    def test_finding_type_is_literal(self) -> None:
        """finding_type must always be 'config_drift' (Literal field)."""
        finding = ConfigDriftFinding(commit_sha="sha1")
        assert finding.finding_type == "config_drift"

    def test_severity_coercion_from_string(self) -> None:
        """Severity coercer normalises string values like 'high' -> Severity.HIGH."""
        finding = ConfigDriftFinding(commit_sha="sha1", severity="medium")  # type: ignore[arg-type]
        assert finding.severity == Severity.MEDIUM

    def test_severity_coercion_fallback(self) -> None:
        """Unknown severity values fall back to HIGH."""
        finding = ConfigDriftFinding(commit_sha="sha1", severity="unknown_value")  # type: ignore[arg-type]
        assert finding.severity == Severity.HIGH

    def test_extra_fields_ignored(self) -> None:
        """extra='ignore' — unexpected fields should not raise."""
        finding = ConfigDriftFinding(commit_sha="sha1", surprise_field="ignored")  # type: ignore[call-arg]
        assert finding.commit_sha == "sha1"

    def test_commit_sha_required(self) -> None:
        """ConfigDriftFinding requires commit_sha."""
        with pytest.raises(Exception):
            ConfigDriftFinding()  # type: ignore[call-arg]


# ── T-11: register_live_tools GitHub injection ────────────────────────────────


def _make_settings(*, github_enabled: bool = True) -> MagicMock:
    settings = MagicMock()
    settings.github.enabled = github_enabled
    settings.github.allowed_repos = []
    return settings


def _make_gke_config() -> MagicMock:
    return MagicMock()


_GKE_PATCHES = [
    "vaig.core.gke.get_gke_credentials",
    "vaig.tools.gke_tools.create_gke_tools",
    "vaig.tools.gcloud_tools.create_gcloud_tools",
    "vaig.tools.plugin_loader.load_all_plugin_tools",
    "vaig.tools.integrations._registry.create_alert_correlation_tools",
]


def _register_live_tools_stub_patches():
    """Context manager patches that prevent network calls in register_live_tools."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("vaig.core.auth.get_gke_credentials", return_value=None),
            patch("vaig.tools.gke_tools.create_gke_tools", return_value=[], create=True),
            patch("vaig.tools.gcloud_tools.create_gcloud_tools", return_value=[], create=True),
            patch("vaig.tools.plugin_loader.load_all_plugin_tools", return_value=[], create=True),
            patch(
                "vaig.tools.integrations._registry.create_alert_correlation_tools",
                return_value=[],
                create=True,
            ),
        ):
            yield

    return _ctx()


class TestRegisterLiveToolsGitHub:
    """Tests for GitHub tool injection in register_live_tools (T-11)."""

    __test__ = True

    def test_github_tools_injected_when_enabled_and_repo_provided(self) -> None:
        """When github.enabled=True and repo is provided, tools are injected."""
        settings = _make_settings(github_enabled=True)
        gke_config = _make_gke_config()

        fake_tool = MagicMock()

        with (
            _register_live_tools_stub_patches(),
            patch(
                "vaig.tools.integrations._github_registry.create_github_repo_tools",
                return_value=[fake_tool],
            ) as mock_create,
        ):
            register_live_tools(
                gke_config,
                settings=settings,
                repo="myorg/myservice",
                repo_ref="main",
            )

            mock_create.assert_called_once_with(settings)

    def test_github_tools_not_injected_when_disabled(self) -> None:
        """When github.enabled=False, no GitHub tools are injected even with --repo."""
        settings = _make_settings(github_enabled=False)
        gke_config = _make_gke_config()

        with (
            _register_live_tools_stub_patches(),
            patch(
                "vaig.tools.integrations._github_registry.create_github_repo_tools",
            ) as mock_create,
        ):
            register_live_tools(
                gke_config,
                settings=settings,
                repo="myorg/myservice",
            )

            mock_create.assert_not_called()

    def test_github_tools_not_injected_without_repo(self) -> None:
        """When repo is None, no GitHub tools are injected even if github is enabled."""
        settings = _make_settings(github_enabled=True)
        gke_config = _make_gke_config()

        with (
            _register_live_tools_stub_patches(),
            patch(
                "vaig.tools.integrations._github_registry.create_github_repo_tools",
            ) as mock_create,
        ):
            register_live_tools(gke_config, settings=settings, repo=None)

            mock_create.assert_not_called()

    def test_repo_ref_defaults_to_head(self) -> None:
        """repo_ref defaults to 'HEAD' when not provided."""
        sig = inspect.signature(register_live_tools)
        assert sig.parameters["repo_ref"].default == "HEAD"

    def test_repo_defaults_to_none(self) -> None:
        """repo defaults to None when not provided."""
        sig = inspect.signature(register_live_tools)
        assert sig.parameters["repo"].default is None


# ── T-12: _planner.py hypothesis types ───────────────────────────────────────


class TestPlannerPromptHypothesisTypes:
    """Tests that commit_correlation and chart_regression are in the planner prompt (T-12)."""

    __test__ = True

    def test_commit_correlation_hint_present(self) -> None:
        """HEALTH_PLANNER_PROMPT contains commit_correlation tool hint."""
        from vaig.skills.service_health.prompts._planner import HEALTH_PLANNER_PROMPT

        assert "commit_correlation" in HEALTH_PLANNER_PROMPT

    def test_chart_regression_hint_present(self) -> None:
        """HEALTH_PLANNER_PROMPT contains chart_regression tool hint."""
        from vaig.skills.service_health.prompts._planner import HEALTH_PLANNER_PROMPT

        assert "chart_regression" in HEALTH_PLANNER_PROMPT

    def test_repo_context_guard_mentioned(self) -> None:
        """Planner prompt notes that repo hints are only used when repo context is available."""
        from vaig.skills.service_health.prompts._planner import HEALTH_PLANNER_PROMPT

        assert "repo context" in HEALTH_PLANNER_PROMPT.lower()


# ── T-10: CLI option forwarding ───────────────────────────────────────────────


class TestLiveCommandRepoForwarding:
    """Tests that --repo and --repo-ref CLI options are forwarded to internal functions (T-10)."""

    __test__ = True

    def test_execute_orchestrated_skill_accepts_repo_params(self) -> None:
        """_execute_orchestrated_skill signature includes repo and repo_ref params."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        sig = inspect.signature(_execute_orchestrated_skill)
        assert "repo" in sig.parameters
        assert "repo_ref" in sig.parameters
        assert sig.parameters["repo"].default is None
        assert sig.parameters["repo_ref"].default == "HEAD"

    def test_execute_live_mode_accepts_repo_params(self) -> None:
        """_execute_live_mode signature includes repo and repo_ref params."""
        from vaig.cli.commands.live import _execute_live_mode

        sig = inspect.signature(_execute_live_mode)
        assert "repo" in sig.parameters
        assert "repo_ref" in sig.parameters
        assert sig.parameters["repo"].default is None
        assert sig.parameters["repo_ref"].default == "HEAD"

    def test_display_dry_run_plan_accepts_repo_params(self) -> None:
        """_display_dry_run_plan signature includes repo and repo_ref params."""
        from vaig.cli.commands.live import _display_dry_run_plan

        sig = inspect.signature(_display_dry_run_plan)
        assert "repo" in sig.parameters
        assert "repo_ref" in sig.parameters

    def test_async_execute_orchestrated_skill_accepts_repo_params(self) -> None:
        """_async_execute_orchestrated_skill signature includes repo and repo_ref params."""
        from vaig.cli.commands.live import _async_execute_orchestrated_skill

        sig = inspect.signature(_async_execute_orchestrated_skill)
        assert "repo" in sig.parameters
        assert "repo_ref" in sig.parameters

"""Tests for runtime config switching — config_switcher module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import (
    GCPConfig,
    GKEConfig,
    ModelsConfig,
    ProjectEntry,
    Settings,
)
from vaig.core.config_switcher import (
    SwitchResult,
    get_config_snapshot,
    switch_cluster,
    switch_location,
    switch_project,
)

# ── Fixtures ─────────────────────────────────────────────────
# _reset_settings is provided by conftest.py (autouse)
# mock_client is provided by conftest.py


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings for config switcher tests."""
    return Settings(
        gcp=GCPConfig(
            project_id="old-project",
            location="us-central1",
            available_projects=[
                ProjectEntry(project_id="old-project", description="Old"),
                ProjectEntry(project_id="known-project", description="Known"),
            ],
        ),
        gke=GKEConfig(
            cluster_name="old-cluster",
            project_id="old-project",
            location="us-central1",
            context="old-context",
        ),
        models=ModelsConfig(default="gemini-2.5-pro"),
    )


# ── SwitchResult ─────────────────────────────────────────────


class TestSwitchResult:
    def test_dataclass_fields(self) -> None:
        result = SwitchResult(
            success=True,
            field="project",
            old_value="old",
            new_value="new",
            message="Switched.",
        )
        assert result.success is True
        assert result.field == "project"
        assert result.old_value == "old"
        assert result.new_value == "new"
        assert result.message == "Switched."
        assert result.reinitialized == []

    def test_reinitialized_default(self) -> None:
        r1 = SwitchResult(success=True, field="x", old_value="", new_value="", message="")
        r2 = SwitchResult(success=True, field="x", old_value="", new_value="", message="")
        # Ensure each instance gets its own list
        r1.reinitialized.append("a")
        assert r2.reinitialized == []


# ── switch_project ───────────────────────────────────────────


class TestSwitchProject:
    def test_empty_project_fails(self, settings: Settings) -> None:
        result = switch_project(settings, "")
        assert result.success is False
        assert "empty" in result.message.lower()
        # Settings must not change
        assert settings.gcp.project_id == "old-project"

    def test_whitespace_only_project_fails(self, settings: Settings) -> None:
        result = switch_project(settings, "   ")
        assert result.success is False

    def test_same_project_noop(self, settings: Settings) -> None:
        result = switch_project(settings, "old-project")
        assert result.success is True
        assert "already" in result.message.lower()

    def test_successful_switch_no_client(self, settings: Settings) -> None:
        result = switch_project(settings, "new-project")
        assert result.success is True
        assert result.old_value == "old-project"
        assert result.new_value == "new-project"
        assert settings.gcp.project_id == "new-project"
        # gke.project_id is NOT synced — the fallback chain in _build_gke_config handles it
        assert settings.gke.project_id == "old-project"
        assert result.reinitialized == []

    def test_successful_switch_with_client(
        self, settings: Settings, mock_client: MagicMock,
    ) -> None:
        result = switch_project(settings, "new-project", client=mock_client)
        assert result.success is True
        mock_client.reinitialize.assert_called_once_with(project="new-project")
        assert "GeminiClient" in result.reinitialized

    def test_client_reinit_failure_rolls_back(
        self, settings: Settings, mock_client: MagicMock,
    ) -> None:
        mock_client.reinitialize.side_effect = Exception("connection failed")
        result = switch_project(settings, "new-project", client=mock_client)
        assert result.success is False
        assert "reinitialize" in result.message.lower()
        # Rollback: gcp.project_id is restored
        assert settings.gcp.project_id == "old-project"
        # gke.project_id is never touched by switch_project — stays as fixture set it
        assert settings.gke.project_id == "old-project"

    def test_unknown_project_warns(self, settings: Settings) -> None:
        result = switch_project(settings, "mystery-project")
        assert result.success is True
        assert "not in available_projects" in result.message

    def test_known_project_no_warning(self, settings: Settings) -> None:
        result = switch_project(settings, "known-project")
        assert result.success is True
        assert "not in available_projects" not in result.message

    def test_no_catalog_no_warning(self) -> None:
        settings = Settings(
            gcp=GCPConfig(project_id="old", available_projects=[]),
        )
        result = switch_project(settings, "any-project")
        assert result.success is True
        assert "not in available_projects" not in result.message

    def test_strips_whitespace(self, settings: Settings) -> None:
        result = switch_project(settings, "  trimmed-project  ")
        assert result.success is True
        assert settings.gcp.project_id == "trimmed-project"


# ── switch_location ──────────────────────────────────────────


class TestSwitchLocation:
    def test_empty_location_fails(self, settings: Settings) -> None:
        result = switch_location(settings, "")
        assert result.success is False
        assert settings.gcp.location == "us-central1"

    def test_same_location_noop(self, settings: Settings) -> None:
        result = switch_location(settings, "us-central1")
        assert result.success is True
        assert "already" in result.message.lower()

    def test_successful_switch_no_client(self, settings: Settings) -> None:
        result = switch_location(settings, "europe-west1")
        assert result.success is True
        assert result.old_value == "us-central1"
        assert result.new_value == "europe-west1"
        assert settings.gcp.location == "europe-west1"

    def test_successful_switch_with_client(
        self, settings: Settings, mock_client: MagicMock,
    ) -> None:
        result = switch_location(settings, "europe-west1", client=mock_client)
        assert result.success is True
        mock_client.reinitialize.assert_called_once_with(location="europe-west1")
        assert "GeminiClient" in result.reinitialized

    def test_client_reinit_failure_rolls_back(
        self, settings: Settings, mock_client: MagicMock,
    ) -> None:
        mock_client.reinitialize.side_effect = Exception("network error")
        result = switch_location(settings, "asia-east1", client=mock_client)
        assert result.success is False
        # Rollback
        assert settings.gcp.location == "us-central1"

    def test_strips_whitespace(self, settings: Settings) -> None:
        result = switch_location(settings, "  europe-west1  ")
        assert result.success is True
        assert settings.gcp.location == "europe-west1"


# ── switch_cluster ───────────────────────────────────────────


class TestSwitchCluster:
    def test_empty_cluster_fails(self, settings: Settings) -> None:
        result = switch_cluster(settings, "")
        assert result.success is False
        assert settings.gke.cluster_name == "old-cluster"

    def test_same_cluster_noop(self, settings: Settings) -> None:
        result = switch_cluster(settings, "old-cluster")
        assert result.success is True
        assert "already" in result.message.lower()

    def test_successful_switch(self, settings: Settings) -> None:
        with patch("vaig.core.config_switcher.clear_k8s_client_cache", create=True):
            result = switch_cluster(settings, "new-cluster")
        assert result.success is True
        assert result.old_value == "old-cluster"
        assert result.new_value == "new-cluster"
        assert settings.gke.cluster_name == "new-cluster"

    def test_switch_with_context(self, settings: Settings) -> None:
        result = switch_cluster(settings, "new-cluster", new_context="new-ctx")
        assert result.success is True
        assert settings.gke.cluster_name == "new-cluster"
        assert settings.gke.context == "new-ctx"

    def test_clears_caches(self, settings: Settings) -> None:
        with (
            patch("vaig.core.config_switcher.clear_k8s_client_cache", create=True) as mock_k8s,
            patch("vaig.core.config_switcher.clear_autopilot_cache", create=True) as mock_ap,
            patch("vaig.core.config_switcher.clear_discovery_cache", create=True) as mock_disc,
        ):
            result = switch_cluster(settings, "new-cluster")
        assert result.success is True
        assert "k8s_client_cache" in result.reinitialized
        assert "autopilot_cache" in result.reinitialized
        assert "discovery_cache" in result.reinitialized

    def test_same_cluster_different_context(self, settings: Settings) -> None:
        result = switch_cluster(settings, "old-cluster", new_context="different-ctx")
        assert result.success is True
        assert settings.gke.context == "different-ctx"

    def test_strips_whitespace(self, settings: Settings) -> None:
        result = switch_cluster(settings, "  trimmed-cluster  ")
        assert result.success is True
        assert settings.gke.cluster_name == "trimmed-cluster"


# ── get_config_snapshot ──────────────────────────────────────


class TestGetConfigSnapshot:
    def test_snapshot_keys(self, settings: Settings) -> None:
        snap = get_config_snapshot(settings)
        assert snap["project"] == "old-project"
        assert snap["location"] == "us-central1"
        assert snap["model"] == "gemini-2.5-pro"
        assert snap["cluster"] == "old-cluster"
        assert snap["context"] == "old-context"
        assert snap["gke_project"] == "old-project"
        assert snap["gke_location"] == "us-central1"

    def test_snapshot_reflects_mutation(self, settings: Settings) -> None:
        switch_project(settings, "new-project")
        snap = get_config_snapshot(settings)
        assert snap["project"] == "new-project"
        # gke_project is NOT synced by switch_project — it stays as configured
        assert snap["gke_project"] == "old-project"

"""Tests for GKEConfig and SkillMetadata.requires_live_tools field."""

from __future__ import annotations

from vaig.core.config import GKEConfig, Settings

# _reset_settings is provided by conftest.py (autouse)


# ── GKEConfig ────────────────────────────────────────────────


class TestGKEConfig:
    def test_defaults(self) -> None:
        cfg = GKEConfig()
        assert cfg.cluster_name == ""
        assert cfg.project_id == ""
        assert cfg.default_namespace == "default"
        assert cfg.kubeconfig_path == ""
        assert cfg.context == ""
        assert cfg.log_limit == 100
        assert cfg.metrics_interval_minutes == 60
        assert cfg.proxy_url == ""

    def test_custom_values(self) -> None:
        cfg = GKEConfig(
            cluster_name="prod-cluster",
            project_id="my-project-123",
            default_namespace="production",
            kubeconfig_path="/home/user/.kube/config",
            context="gke_my-project_us-central1_prod",
            log_limit=500,
            metrics_interval_minutes=120,
            proxy_url="https://proxy.example.com:8443",
        )
        assert cfg.cluster_name == "prod-cluster"
        assert cfg.project_id == "my-project-123"
        assert cfg.default_namespace == "production"
        assert cfg.kubeconfig_path == "/home/user/.kube/config"
        assert cfg.context == "gke_my-project_us-central1_prod"
        assert cfg.log_limit == 500
        assert cfg.metrics_interval_minutes == 120
        assert cfg.proxy_url == "https://proxy.example.com:8443"

    def test_partial_values(self) -> None:
        cfg = GKEConfig(cluster_name="dev-cluster")
        assert cfg.cluster_name == "dev-cluster"
        assert cfg.project_id == ""  # Other defaults preserved
        assert cfg.default_namespace == "default"

    def test_serialization_round_trip(self) -> None:
        cfg = GKEConfig(
            cluster_name="test",
            project_id="proj",
            log_limit=200,
        )
        data = cfg.model_dump()
        cfg2 = GKEConfig(**data)
        assert cfg2.cluster_name == cfg.cluster_name
        assert cfg2.project_id == cfg.project_id
        assert cfg2.log_limit == cfg.log_limit


# ── GKEConfig in Settings ───────────────────────────────────


class TestGKEConfigInSettings:
    def test_settings_has_gke_field(self) -> None:
        settings = Settings()
        assert hasattr(settings, "gke")
        assert isinstance(settings.gke, GKEConfig)

    def test_settings_gke_defaults(self) -> None:
        settings = Settings()
        assert settings.gke.cluster_name == ""
        assert settings.gke.default_namespace == "default"
        assert settings.gke.log_limit == 100


# ── SkillMetadata.requires_live_tools ────────────────────────


class TestSkillMetadataRequiresLiveTools:
    def test_default_is_false(self) -> None:
        from vaig.skills.base import SkillMetadata

        meta = SkillMetadata(
            name="test-skill",
            display_name="Test Skill",
            description="A test skill.",
        )
        assert meta.requires_live_tools is False

    def test_can_set_to_true(self) -> None:
        from vaig.skills.base import SkillMetadata

        meta = SkillMetadata(
            name="live-skill",
            display_name="Live Skill",
            description="Requires live tools.",
            requires_live_tools=True,
        )
        assert meta.requires_live_tools is True

    def test_field_is_bool(self) -> None:
        from vaig.skills.base import SkillMetadata

        meta = SkillMetadata(
            name="skill",
            display_name="Skill",
            description="desc",
        )
        assert isinstance(meta.requires_live_tools, bool)

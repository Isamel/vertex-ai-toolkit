"""Tests for core configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.config import (
    AuthConfig,
    AuthMode,
    ContextConfig,
    GCPConfig,
    GenerationConfig,
    ModelInfo,
    ModelsConfig,
    SessionConfig,
    Settings,
    SkillsConfig,
    reset_settings,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset the settings singleton between tests."""
    reset_settings()


class TestAuthMode:
    def test_adc_value(self) -> None:
        assert AuthMode.ADC == "adc"

    def test_impersonate_value(self) -> None:
        assert AuthMode.IMPERSONATE == "impersonate"


class TestGCPConfig:
    def test_defaults(self) -> None:
        cfg = GCPConfig()
        assert cfg.project_id == ""
        assert cfg.location == "us-central1"

    def test_custom_values(self) -> None:
        cfg = GCPConfig(project_id="my-project", location="europe-west1")
        assert cfg.project_id == "my-project"
        assert cfg.location == "europe-west1"


class TestAuthConfig:
    def test_defaults(self) -> None:
        cfg = AuthConfig()
        assert cfg.mode == AuthMode.ADC
        assert cfg.impersonate_sa == ""

    def test_impersonate_mode(self) -> None:
        cfg = AuthConfig(mode=AuthMode.IMPERSONATE, impersonate_sa="sa@proj.iam.gserviceaccount.com")
        assert cfg.mode == AuthMode.IMPERSONATE
        assert "sa@proj" in cfg.impersonate_sa


class TestGenerationConfig:
    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 8192
        assert cfg.top_p == 0.95
        assert cfg.top_k == 40

    def test_custom(self) -> None:
        cfg = GenerationConfig(temperature=0.0, max_output_tokens=1024)
        assert cfg.temperature == 0.0
        assert cfg.max_output_tokens == 1024


class TestModelsConfig:
    def test_defaults(self) -> None:
        cfg = ModelsConfig()
        assert cfg.default == "gemini-2.5-pro"
        assert cfg.fallback == "gemini-2.5-flash"
        assert cfg.available == []

    def test_with_available_models(self) -> None:
        models = [
            ModelInfo(id="gemini-2.5-pro", description="Pro"),
            ModelInfo(id="gemini-2.5-flash", description="Flash"),
        ]
        cfg = ModelsConfig(available=models)
        assert len(cfg.available) == 2
        assert cfg.available[0].id == "gemini-2.5-pro"


class TestSkillsConfig:
    def test_defaults(self) -> None:
        cfg = SkillsConfig()
        assert "rca" in cfg.enabled
        assert "anomaly" in cfg.enabled
        assert "migration" in cfg.enabled
        assert cfg.custom_dir is None

    def test_custom_dir(self) -> None:
        cfg = SkillsConfig(custom_dir="~/.vaig/skills")
        assert cfg.custom_dir == "~/.vaig/skills"


class TestContextConfig:
    def test_defaults(self) -> None:
        cfg = ContextConfig()
        assert cfg.max_file_size_mb == 50
        assert cfg.supported_extensions == {}
        assert cfg.ignore_patterns == []

    def test_custom(self) -> None:
        cfg = ContextConfig(
            max_file_size_mb=10,
            supported_extensions={"code": [".py", ".js"]},
            ignore_patterns=["__pycache__"],
        )
        assert cfg.max_file_size_mb == 10
        assert ".py" in cfg.supported_extensions["code"]


class TestSettings:
    def test_default_construction(self) -> None:
        s = Settings()
        assert s.gcp.location == "us-central1"
        assert s.models.default == "gemini-2.5-pro"
        assert s.auth.mode == AuthMode.ADC
        assert s.session.auto_save is True

    def test_current_model(self) -> None:
        s = Settings()
        assert s.current_model == "gemini-2.5-pro"

    def test_db_path_resolved(self) -> None:
        s = Settings()
        path = s.db_path_resolved
        assert isinstance(path, Path)
        assert "~" not in str(path)  # Should be expanded

    def test_get_model_info_found(self) -> None:
        s = Settings(
            models=ModelsConfig(
                available=[ModelInfo(id="gemini-2.5-pro", description="Pro model")]
            )
        )
        info = s.get_model_info("gemini-2.5-pro")
        assert info is not None
        assert info.description == "Pro model"

    def test_get_model_info_not_found(self) -> None:
        s = Settings()
        info = s.get_model_info("nonexistent")
        assert info is None

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "gcp:\n  project_id: test-project\n  location: europe-west1\n"
        )
        s = Settings.load(config_file)
        assert s.gcp.project_id == "test-project"
        assert s.gcp.location == "europe-west1"

    def test_load_no_yaml_uses_defaults(self) -> None:
        s = Settings.load("/nonexistent/path.yaml")
        assert s.gcp.project_id == ""
        assert s.models.default == "gemini-2.5-pro"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAIG_GCP__PROJECT_ID", "env-project")
        s = Settings()
        assert s.gcp.project_id == "env-project"

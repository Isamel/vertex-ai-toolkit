"""Tests for platform configuration and service container integration (Phase 1).

Covers:
  - SC-AUTH-001a: Platform disabled by default
  - SC-AUTH-001b: Platform config from YAML
  - SC-AUTH-001c: Platform config from env vars
  - SC-AUTH-006a: Container without platform
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from vaig.core.config import PlatformConfig, Settings
from vaig.core.container import ServiceContainer, build_container
from vaig.core.event_bus import EventBus


class TestPlatformConfigDefaults:
    """SC-AUTH-001a: Platform disabled by default."""

    def test_platform_disabled_by_default(self) -> None:
        """A fresh Settings with no overrides has platform.enabled = False."""
        settings = Settings()
        assert settings.platform.enabled is False

    def test_platform_backend_url_empty_by_default(self) -> None:
        settings = Settings()
        assert settings.platform.backend_url == ""

    def test_platform_org_id_empty_by_default(self) -> None:
        settings = Settings()
        assert settings.platform.org_id == ""

    def test_platform_config_model_defaults(self) -> None:
        cfg = PlatformConfig()
        assert cfg.enabled is False
        assert cfg.backend_url == ""
        assert cfg.org_id == ""


class TestPlatformConfigYAML:
    """SC-AUTH-001b: Platform config from YAML."""

    def test_platform_enabled_from_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "platform": {
                "enabled": True,
                "backend_url": "https://api.example.com",
                "org_id": "my-org",
            }
        }
        config_file = tmp_path / "vaig.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        settings = Settings.load(config_path=str(config_file))

        assert settings.platform.enabled is True
        assert settings.platform.backend_url == "https://api.example.com"
        assert settings.platform.org_id == "my-org"

    def test_platform_partial_yaml(self, tmp_path: Path) -> None:
        """Only backend_url set — enabled still False, org_id empty."""
        config_data = {
            "platform": {
                "backend_url": "https://api.example.com",
            }
        }
        config_file = tmp_path / "vaig.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        settings = Settings.load(config_path=str(config_file))

        assert settings.platform.enabled is False
        assert settings.platform.backend_url == "https://api.example.com"
        assert settings.platform.org_id == ""

    def test_settings_load_without_platform_section(self, tmp_path: Path) -> None:
        """Settings.load() must NOT break if platform: section is missing."""
        config_data = {"gcp": {"project_id": "test-project"}}
        config_file = tmp_path / "vaig.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        settings = Settings.load(config_path=str(config_file))

        assert settings.platform.enabled is False
        assert settings.gcp.project_id == "test-project"


class TestPlatformConfigEnvVars:
    """SC-AUTH-001c: Platform config from env vars."""

    def test_platform_enabled_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAIG_PLATFORM__ENABLED", "true")
        monkeypatch.setenv("VAIG_PLATFORM__BACKEND_URL", "https://api.example.com")
        monkeypatch.setenv("VAIG_PLATFORM__ORG_ID", "env-org")

        settings = Settings()

        assert settings.platform.enabled is True
        assert settings.platform.backend_url == "https://api.example.com"
        assert settings.platform.org_id == "env-org"

    def test_platform_env_supplies_missing_yaml_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars supply fields absent from YAML (e.g. org_id).

        Note: ``Settings.load()`` passes YAML values as init kwargs to
        pydantic-settings, which treats them as higher priority than env
        vars.  This means env vars cannot override a value that is
        explicitly present in any YAML layer (including default.yaml).
        They *can* supply values for keys not set in YAML at all.
        """
        config_data = {
            "platform": {
                "enabled": True,
                "backend_url": "https://yaml.example.com",
            }
        }
        config_file = tmp_path / "vaig.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        monkeypatch.setenv("VAIG_PLATFORM__ORG_ID", "env-org")

        settings = Settings.load(config_path=str(config_file))

        assert settings.platform.enabled is True
        assert settings.platform.backend_url == "https://yaml.example.com"
        assert settings.platform.org_id == "env-org"


class TestContainerWithoutPlatform:
    """SC-AUTH-006a: Container without platform auth."""

    def test_container_platform_auth_none_by_default(self) -> None:
        """ServiceContainer has platform_auth=None when not provided."""
        container = ServiceContainer(
            settings=Settings(),
            gemini_client=MagicMock(),
            k8s_provider=None,
            gcp_provider=None,
            event_bus=EventBus.get(),
        )
        assert container.platform_auth is None

    def test_build_container_platform_auth_none_when_disabled(self) -> None:
        """build_container() sets platform_auth=None when platform.enabled=False."""
        settings = Settings()
        assert settings.platform.enabled is False

        container = build_container(settings)

        assert container.platform_auth is None

    def test_container_frozen_with_platform_auth(self) -> None:
        """ServiceContainer remains frozen — platform_auth is read-only."""
        container = ServiceContainer(
            settings=Settings(),
            gemini_client=MagicMock(),
            k8s_provider=None,
            gcp_provider=None,
            event_bus=EventBus.get(),
            platform_auth=MagicMock(),
        )
        assert container.platform_auth is not None

        with pytest.raises(AttributeError):
            container.platform_auth = None  # type: ignore[misc]

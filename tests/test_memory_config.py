"""Tests for MemoryConfig in Settings."""
from __future__ import annotations

from vaig.core.config import MemoryConfig, Settings


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.enabled is False
        assert "~/.vaig/memory" in cfg.store_path
        assert cfg.recurrence_threshold == 2
        assert cfg.chronic_threshold == 5
        assert cfg.max_age_days == 90

    def test_settings_has_memory_field(self) -> None:
        settings = Settings()
        assert hasattr(settings, "memory")
        assert isinstance(settings.memory, MemoryConfig)

    def test_settings_memory_disabled_by_default(self) -> None:
        settings = Settings()
        assert settings.memory.enabled is False

    def test_memory_config_custom_values(self) -> None:
        cfg = MemoryConfig(
            enabled=True,
            store_path="/tmp/test-memory",
            recurrence_threshold=3,
            chronic_threshold=10,
            max_age_days=30,
        )
        assert cfg.enabled is True
        assert cfg.store_path == "/tmp/test-memory"
        assert cfg.recurrence_threshold == 3
        assert cfg.chronic_threshold == 10
        assert cfg.max_age_days == 30

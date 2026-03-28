"""Tests for core configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.config import (
    AuthConfig,
    AuthMode,
    BudgetConfig,
    ContextConfig,
    GCPConfig,
    GenerationConfig,
    LoggingConfig,
    MCPConfig,
    ModelInfo,
    ModelsConfig,
    PluginConfig,
    ProjectEntry,
    SafetyConfig,
    SafetySettingConfig,
    Settings,
    SkillsConfig,
    _deep_merge,
    _strip_empty_strings,
    _strip_empty_strings_in_list,
)

# _reset_settings is provided by conftest.py (autouse)


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
        assert cfg.available_projects == []

    def test_custom_values(self) -> None:
        cfg = GCPConfig(project_id="my-project", location="europe-west1")
        assert cfg.project_id == "my-project"
        assert cfg.location == "europe-west1"

    def test_available_projects(self) -> None:
        projects = [
            ProjectEntry(project_id="proj-a", description="Vertex AI", role="vertex-ai"),
            ProjectEntry(project_id="proj-b", description="GKE infra", role="gke"),
        ]
        cfg = GCPConfig(available_projects=projects)
        assert len(cfg.available_projects) == 2
        assert cfg.available_projects[0].project_id == "proj-a"
        assert cfg.available_projects[1].role == "gke"


class TestProjectEntry:
    """Tests for ProjectEntry model."""

    def test_minimal(self) -> None:
        entry = ProjectEntry(project_id="my-project")
        assert entry.project_id == "my-project"
        assert entry.description == ""
        assert entry.role == ""

    def test_full(self) -> None:
        entry = ProjectEntry(
            project_id="my-project",
            description="Production Vertex AI",
            role="vertex-ai",
        )
        assert entry.project_id == "my-project"
        assert entry.description == "Production Vertex AI"
        assert entry.role == "vertex-ai"

    def test_requires_project_id(self) -> None:
        with pytest.raises(Exception):
            ProjectEntry()  # type: ignore[call-arg]

    def test_role_values(self) -> None:
        for role in ("vertex-ai", "gke", "both", ""):
            entry = ProjectEntry(project_id="p", role=role)
            assert entry.role == role

    def test_serialization_round_trip(self) -> None:
        entry = ProjectEntry(
            project_id="proj-1",
            description="Test project",
            role="both",
        )
        data = entry.model_dump()
        restored = ProjectEntry(**data)
        assert restored.project_id == entry.project_id
        assert restored.description == entry.description
        assert restored.role == entry.role

    def test_available_projects_in_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.gcp.available_projects == []

    def test_available_projects_from_yaml_data(self) -> None:
        settings = Settings(
            gcp={  # type: ignore[arg-type]
                "project_id": "main",
                "available_projects": [
                    {"project_id": "proj-a", "description": "Vertex AI", "role": "vertex-ai"},
                    {"project_id": "proj-b"},
                ],
            },
        )
        assert len(settings.gcp.available_projects) == 2
        assert settings.gcp.available_projects[0].project_id == "proj-a"
        assert settings.gcp.available_projects[0].description == "Vertex AI"
        assert settings.gcp.available_projects[1].project_id == "proj-b"
        assert settings.gcp.available_projects[1].description == ""

    def test_available_projects_from_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "gcp:\n"
            "  project_id: test-proj\n"
            "  available_projects:\n"
            "    - project_id: alpha\n"
            "      description: Alpha project\n"
            "      role: vertex-ai\n"
            "    - project_id: beta\n"
            "      role: gke\n"
        )
        s = Settings.load(config_file)
        assert s.gcp.project_id == "test-proj"
        assert len(s.gcp.available_projects) == 2
        assert s.gcp.available_projects[0].project_id == "alpha"
        assert s.gcp.available_projects[0].role == "vertex-ai"
        assert s.gcp.available_projects[1].project_id == "beta"
        assert s.gcp.available_projects[1].description == ""

    def test_backward_compat_without_available_projects(self) -> None:
        """Existing configs without available_projects should use empty list."""
        settings = Settings()
        assert settings.gcp.available_projects == []


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
        assert cfg.max_output_tokens == 16384
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

    def test_logging_defaults_in_settings(self) -> None:
        s = Settings()
        assert s.logging.level == "WARNING"
        assert s.logging.show_path is False

    def test_logging_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "logging:\n  level: DEBUG\n  show_path: true\n"
        )
        s = Settings.load(config_file)
        assert s.logging.level == "DEBUG"
        assert s.logging.show_path is True

    def test_logging_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAIG_LOGGING__LEVEL", "ERROR")
        s = Settings()
        assert s.logging.level == "ERROR"


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level == "WARNING"
        assert cfg.show_path is False

    def test_custom_level(self) -> None:
        cfg = LoggingConfig(level="DEBUG")
        assert cfg.level == "DEBUG"

    def test_custom_show_path(self) -> None:
        cfg = LoggingConfig(show_path=True)
        assert cfg.show_path is True

    def test_all_levels_accepted(self) -> None:
        """LoggingConfig should accept any string — validation happens at setup time."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = LoggingConfig(level=level)
            assert cfg.level == level


class TestStripEmptyStrings:
    """Tests for _strip_empty_strings and _strip_empty_strings_in_list helpers."""

    def test_removes_top_level_empty_strings(self) -> None:
        data = {"project_id": "", "location": "global", "name": ""}
        result = _strip_empty_strings(data)
        assert result == {"location": "global"}

    def test_preserves_non_empty_values(self) -> None:
        data = {"a": "hello", "b": 42, "c": True, "d": 0, "e": False}
        result = _strip_empty_strings(data)
        assert result == data

    def test_recurses_into_nested_dicts(self) -> None:
        data = {"gcp": {"project_id": "", "location": "global"}}
        result = _strip_empty_strings(data)
        assert result == {"gcp": {"location": "global"}}

    def test_removes_empty_nested_dict(self) -> None:
        """If all values in a nested dict are empty strings, the key is removed."""
        data = {"gcp": {"project_id": "", "location": ""}, "name": "test"}
        result = _strip_empty_strings(data)
        assert result == {"name": "test"}

    def test_recurses_into_list_of_dicts(self) -> None:
        """The critical bug fix — models.available has list of dicts with empty strings."""
        data = {
            "models": {
                "default": "gemini-2.5-pro",
                "available": [
                    {"id": "gemini-2.5-pro", "description": ""},
                    {"id": "gemini-2.5-flash", "description": "Fast model"},
                ],
            }
        }
        result = _strip_empty_strings(data)
        assert result["models"]["default"] == "gemini-2.5-pro"
        available = result["models"]["available"]
        assert len(available) == 2
        # First item: description was "" so it's stripped
        assert available[0] == {"id": "gemini-2.5-pro"}
        # Second item: description preserved
        assert available[1] == {"id": "gemini-2.5-flash", "description": "Fast model"}

    def test_list_of_plain_strings_filters_empty(self) -> None:
        data = {"tags": ["python", "", "cli", ""]}
        result = _strip_empty_strings(data)
        assert result == {"tags": ["python", "cli"]}

    def test_nested_lists(self) -> None:
        data = {"matrix": [["a", "", "b"], ["", "c"]]}
        result = _strip_empty_strings(data)
        assert result == {"matrix": [["a", "b"], ["c"]]}

    def test_mixed_list_content(self) -> None:
        """Lists with dicts, strings, and numbers."""
        data = {
            "items": [
                {"name": "test", "value": ""},
                "keep",
                "",
                42,
                {"empty": ""},
            ]
        }
        result = _strip_empty_strings(data)
        items = result["items"]
        assert items[0] == {"name": "test"}
        assert items[1] == "keep"
        assert items[2] == 42
        # {"empty": ""} becomes {} — empty dict stays in list (not filtered)
        assert items[3] == {}

    def test_empty_input(self) -> None:
        assert _strip_empty_strings({}) == {}

    def test_all_empty_strings(self) -> None:
        data = {"a": "", "b": "", "c": ""}
        assert _strip_empty_strings(data) == {}

    def test_preserves_none_values(self) -> None:
        """None is not an empty string and should be preserved."""
        data = {"a": None, "b": "", "c": "valid"}
        result = _strip_empty_strings(data)
        assert result == {"a": None, "c": "valid"}

    def test_preserves_zero_and_false(self) -> None:
        """0 and False should not be stripped (they're not empty strings)."""
        data = {"count": 0, "enabled": False, "name": ""}
        result = _strip_empty_strings(data)
        assert result == {"count": 0, "enabled": False}

    def test_deeply_nested_structure(self) -> None:
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "keep": "yes",
                        "remove": "",
                    }
                }
            }
        }
        result = _strip_empty_strings(data)
        assert result == {"level1": {"level2": {"level3": {"keep": "yes"}}}}

    def test_realistic_yaml_config(self) -> None:
        """Simulates a realistic YAML config with empty strings throughout."""
        data = {
            "gcp": {"project_id": "", "location": "global"},
            "auth": {"mode": "adc", "impersonate_sa": ""},
            "models": {
                "default": "gemini-2.5-pro",
                "fallback": "",
                "available": [
                    {"id": "gemini-2.5-pro", "description": "Pro model"},
                    {"id": "gemini-2.5-flash", "description": ""},
                    {"id": "gemini-3.1-pro-preview", "description": ""},
                ],
            },
            "session": {"db_path": "", "auto_save": True},
        }
        result = _strip_empty_strings(data)
        # project_id removed, location kept
        assert result["gcp"] == {"location": "global"}
        # impersonate_sa removed
        assert result["auth"] == {"mode": "adc"}
        # fallback removed, available cleaned
        assert result["models"]["default"] == "gemini-2.5-pro"
        assert "fallback" not in result["models"]
        assert result["models"]["available"][0] == {"id": "gemini-2.5-pro", "description": "Pro model"}
        assert result["models"]["available"][1] == {"id": "gemini-2.5-flash"}
        assert result["models"]["available"][2] == {"id": "gemini-3.1-pro-preview"}
        # db_path removed, auto_save kept
        assert result["session"] == {"auto_save": True}


class TestStripEmptyStringsInList:
    """Direct tests for the list helper function."""

    def test_empty_list(self) -> None:
        assert _strip_empty_strings_in_list([]) == []

    def test_all_empty_strings(self) -> None:
        assert _strip_empty_strings_in_list(["", "", ""]) == []

    def test_preserves_non_strings(self) -> None:
        result = _strip_empty_strings_in_list([1, 2.5, True, None])
        assert result == [1, 2.5, True, None]

    def test_filters_only_empty_strings(self) -> None:
        result = _strip_empty_strings_in_list(["a", "", "b", "", "c"])
        assert result == ["a", "b", "c"]

    def test_recurses_into_dicts(self) -> None:
        result = _strip_empty_strings_in_list([{"key": "val", "empty": ""}])
        assert result == [{"key": "val"}]

    def test_recurses_into_nested_lists(self) -> None:
        result = _strip_empty_strings_in_list([["a", ""], ["", "b"]])
        assert result == [["a"], ["b"]]


class TestDeepMerge:
    """Tests for _deep_merge helper function."""

    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_adds_new_keys(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_nested_dict_is_merged_not_replaced(self) -> None:
        base = {"logging": {"level": "WARNING", "file_enabled": False}}
        override = {"logging": {"file_enabled": True}}
        result = _deep_merge(base, override)
        # file_enabled is overridden, level is preserved
        assert result == {"logging": {"level": "WARNING", "file_enabled": True}}

    def test_non_dict_value_replaces_entirely(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        assert _deep_merge(base, override) == {"items": [4, 5]}

    def test_empty_override(self) -> None:
        base = {"a": 1, "b": 2}
        assert _deep_merge(base, {}) == {"a": 1, "b": 2}

    def test_empty_base(self) -> None:
        override = {"a": 1}
        assert _deep_merge({}, override) == {"a": 1}

    def test_does_not_mutate_base(self) -> None:
        base = {"logging": {"level": "WARNING"}}
        override = {"logging": {"level": "DEBUG"}}
        _deep_merge(base, override)
        assert base["logging"]["level"] == "WARNING"

    def test_deeply_nested(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}


class TestSettingsLoadMerging:
    """Tests verifying that Settings.load() deep-merges all config files."""

    def test_user_config_overrides_project_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core bug fix: user config (~/.vaig/config.yaml) must override the
        project's default config (config/default.yaml) rather than being ignored."""
        # Set up a project default config with file_enabled: false
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "logging:\n  file_enabled: false\n  level: WARNING\n"
        )

        # Set up a user home config with file_enabled: true
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        vaig_dir = home_dir / ".vaig"
        vaig_dir.mkdir()
        (vaig_dir / "config.yaml").write_text(
            "logging:\n  file_enabled: true\n"
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setenv("HOME", str(home_dir))
        # Ensure Path.home() returns our fake home directory
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home_dir))

        s = Settings.load()
        # User config should have enabled file logging
        assert s.logging.file_enabled is True
        # Other logging settings from the default should still be present
        assert s.logging.level == "WARNING"

    def test_project_vaig_yaml_overrides_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """vaig.yaml in cwd should have higher priority than ~/.vaig/config.yaml."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        home_dir = tmp_path / "home"
        home_dir.mkdir()
        vaig_dir = home_dir / ".vaig"
        vaig_dir.mkdir()
        (vaig_dir / "config.yaml").write_text(
            "logging:\n  level: DEBUG\n"
        )
        (project_dir / "vaig.yaml").write_text(
            "logging:\n  level: ERROR\n"
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home_dir))

        s = Settings.load()
        assert s.logging.level == "ERROR"

    def test_explicit_config_path_highest_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit config_path should take priority over all other configs."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        home_dir = tmp_path / "home"
        home_dir.mkdir()
        vaig_dir = home_dir / ".vaig"
        vaig_dir.mkdir()
        (vaig_dir / "config.yaml").write_text(
            "logging:\n  level: INFO\n"
        )

        explicit_config = tmp_path / "explicit.yaml"
        explicit_config.write_text("logging:\n  level: CRITICAL\n")

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home_dir))

        s = Settings.load(config_path=explicit_config)
        assert s.logging.level == "CRITICAL"

    def test_non_overlapping_keys_from_multiple_configs_are_all_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keys set only in a lower-priority config should survive in the merged result."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "logging:\n  level: WARNING\n  file_enabled: false\n"
        )

        home_dir = tmp_path / "home"
        home_dir.mkdir()
        vaig_dir = home_dir / ".vaig"
        vaig_dir.mkdir()
        # User only overrides file_enabled, leaves level untouched
        (vaig_dir / "config.yaml").write_text(
            "logging:\n  file_enabled: true\n"
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home_dir))

        s = Settings.load()
        assert s.logging.file_enabled is True
        assert s.logging.level == "WARNING"  # preserved from default


# ══════════════════════════════════════════════════════════════
# BudgetConfig
# ══════════════════════════════════════════════════════════════


class TestBudgetConfig:
    """Tests for BudgetConfig validation and defaults."""

    def test_defaults(self) -> None:
        config = BudgetConfig()
        assert config.enabled is False
        assert config.max_cost_usd == 5.0
        assert config.warn_threshold == 0.8
        assert config.action == "warn"

    def test_enabled_flag(self) -> None:
        config = BudgetConfig(enabled=True)
        assert config.enabled is True

    def test_custom_max_cost(self) -> None:
        config = BudgetConfig(max_cost_usd=25.0)
        assert config.max_cost_usd == 25.0

    def test_custom_warn_threshold(self) -> None:
        config = BudgetConfig(warn_threshold=0.5)
        assert config.warn_threshold == 0.5

    def test_action_warn(self) -> None:
        config = BudgetConfig(action="warn")
        assert config.action == "warn"

    def test_action_stop(self) -> None:
        config = BudgetConfig(action="stop")
        assert config.action == "stop"

    def test_action_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            BudgetConfig(action="invalid")  # type: ignore[arg-type]

    def test_budget_in_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.budget.enabled is False
        assert settings.budget.max_cost_usd == 5.0

    def test_full_config(self) -> None:
        config = BudgetConfig(
            enabled=True,
            max_cost_usd=50.0,
            warn_threshold=0.9,
            action="stop",
        )
        assert config.enabled is True
        assert config.max_cost_usd == 50.0
        assert config.warn_threshold == 0.9
        assert config.action == "stop"


# ══════════════════════════════════════════════════════════════
# MCPConfig — auto_register field
# ══════════════════════════════════════════════════════════════


class TestMCPConfigAutoRegister:
    """Tests for the auto_register field added to MCPConfig."""

    def test_default_auto_register_is_false(self) -> None:
        cfg = MCPConfig()
        assert cfg.auto_register is False

    def test_auto_register_enabled(self) -> None:
        cfg = MCPConfig(auto_register=True)
        assert cfg.auto_register is True

    def test_auto_register_in_settings_default(self) -> None:
        settings = Settings()
        assert settings.mcp.auto_register is False

    def test_auto_register_from_yaml_data(self) -> None:
        settings = Settings(
            mcp={"enabled": True, "auto_register": True, "servers": []},  # type: ignore[arg-type]
        )
        assert settings.mcp.enabled is True
        assert settings.mcp.auto_register is True

    def test_backward_compat_without_auto_register(self) -> None:
        """Existing configs without auto_register should still work."""
        cfg = MCPConfig(enabled=True)
        assert cfg.enabled is True
        assert cfg.auto_register is False


# ══════════════════════════════════════════════════════════════
# PluginConfig
# ══════════════════════════════════════════════════════════════


class TestPluginConfig:
    """Tests for the new PluginConfig model."""

    def test_defaults(self) -> None:
        cfg = PluginConfig()
        assert cfg.enabled is False
        assert cfg.directories == []

    def test_enabled(self) -> None:
        cfg = PluginConfig(enabled=True)
        assert cfg.enabled is True

    def test_with_directories(self) -> None:
        cfg = PluginConfig(enabled=True, directories=["./plugins", "~/.vaig/plugins"])
        assert cfg.enabled is True
        assert len(cfg.directories) == 2
        assert "./plugins" in cfg.directories
        assert "~/.vaig/plugins" in cfg.directories

    def test_empty_directories(self) -> None:
        cfg = PluginConfig(enabled=True, directories=[])
        assert cfg.directories == []

    def test_plugins_in_settings_defaults(self) -> None:
        settings = Settings()
        assert hasattr(settings, "plugins")
        assert isinstance(settings.plugins, PluginConfig)
        assert settings.plugins.enabled is False
        assert settings.plugins.directories == []

    def test_plugins_from_yaml_data(self) -> None:
        settings = Settings(
            plugins={"enabled": True, "directories": ["./my-plugins"]},  # type: ignore[arg-type]
        )
        assert settings.plugins.enabled is True
        assert settings.plugins.directories == ["./my-plugins"]

    def test_backward_compat_without_plugins(self) -> None:
        """Existing configs without plugins section should use defaults."""
        settings = Settings()
        assert settings.plugins.enabled is False
        assert settings.plugins.directories == []


# ══════════════════════════════════════════════════════════════
# SafetyConfig / SafetySettingConfig
# ══════════════════════════════════════════════════════════════


class TestSafetySettingConfig:
    """Tests for SafetySettingConfig model."""

    def test_requires_category(self) -> None:
        cfg = SafetySettingConfig(category="HARM_CATEGORY_HARASSMENT")
        assert cfg.category == "HARM_CATEGORY_HARASSMENT"
        assert cfg.threshold == "BLOCK_MEDIUM_AND_ABOVE"

    def test_custom_threshold(self) -> None:
        cfg = SafetySettingConfig(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_ONLY_HIGH",
        )
        assert cfg.category == "HARM_CATEGORY_HATE_SPEECH"
        assert cfg.threshold == "BLOCK_ONLY_HIGH"

    def test_block_none_threshold(self) -> None:
        cfg = SafetySettingConfig(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        )
        assert cfg.threshold == "BLOCK_NONE"


class TestSafetyConfig:
    """Tests for SafetyConfig model."""

    def test_defaults(self) -> None:
        cfg = SafetyConfig()
        assert cfg.enabled is True
        assert cfg.settings == []

    def test_disabled(self) -> None:
        cfg = SafetyConfig(enabled=False)
        assert cfg.enabled is False

    def test_with_settings(self) -> None:
        cfg = SafetyConfig(
            enabled=True,
            settings=[
                SafetySettingConfig(category="HARM_CATEGORY_HARASSMENT"),
                SafetySettingConfig(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
            ],
        )
        assert len(cfg.settings) == 2
        assert cfg.settings[0].category == "HARM_CATEGORY_HARASSMENT"
        assert cfg.settings[0].threshold == "BLOCK_MEDIUM_AND_ABOVE"
        assert cfg.settings[1].threshold == "BLOCK_LOW_AND_ABOVE"

    def test_safety_in_settings_defaults(self) -> None:
        settings = Settings()
        assert hasattr(settings, "safety")
        assert isinstance(settings.safety, SafetyConfig)
        assert settings.safety.enabled is True
        assert settings.safety.settings == []

    def test_safety_from_yaml_data(self) -> None:
        settings = Settings(
            safety={  # type: ignore[arg-type]
                "enabled": True,
                "settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                ],
            },
        )
        assert settings.safety.enabled is True
        assert len(settings.safety.settings) == 1
        assert settings.safety.settings[0].category == "HARM_CATEGORY_HARASSMENT"
        assert settings.safety.settings[0].threshold == "BLOCK_ONLY_HIGH"

    def test_backward_compat_without_safety(self) -> None:
        """Existing configs without safety section should use defaults."""
        settings = Settings()
        assert settings.safety.enabled is True
        assert settings.safety.settings == []

    def test_safety_from_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "safety:\n"
            "  enabled: true\n"
            "  settings:\n"
            "    - category: HARM_CATEGORY_HARASSMENT\n"
            "      threshold: BLOCK_LOW_AND_ABOVE\n"
            "    - category: HARM_CATEGORY_HATE_SPEECH\n"
        )
        s = Settings.load(config_file)
        assert s.safety.enabled is True
        assert len(s.safety.settings) == 2
        assert s.safety.settings[0].threshold == "BLOCK_LOW_AND_ABOVE"
        # Second entry uses default threshold
        assert s.safety.settings[1].threshold == "BLOCK_MEDIUM_AND_ABOVE"


# ── Language field tests ──────────────────────────────────────


class TestSettingsLanguage:
    """Settings.language field — default and explicit override."""

    def test_default_language_is_english(self) -> None:
        s = Settings()
        assert s.language == "en"

    def test_custom_language_value(self) -> None:
        s = Settings(language="es")
        assert s.language == "es"

    def test_language_accepts_bcp47_codes(self) -> None:
        for code in ("pt", "ja", "de", "fr"):
            s = Settings(language=code)
            assert s.language == code

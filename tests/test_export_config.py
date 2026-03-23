"""Tests for ExportConfig Pydantic model."""

from __future__ import annotations

from vaig.core.config import ExportConfig, Settings


class TestExportConfigDefaults:
    """Verify all default values match the spec."""

    def test_enabled_is_false_by_default(self) -> None:
        """Safety check — export must be opt-in, never auto-enabled."""
        cfg = ExportConfig()
        assert cfg.enabled is False

    def test_bigquery_dataset_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.bigquery_dataset == "vaig_analytics"

    def test_gcp_project_id_default_is_empty(self) -> None:
        cfg = ExportConfig()
        assert cfg.gcp_project_id == ""

    def test_auto_export_reports_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.auto_export_reports is False

    def test_auto_export_telemetry_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.auto_export_telemetry is False

    def test_gcs_bucket_default_is_empty(self) -> None:
        cfg = ExportConfig()
        assert cfg.gcs_bucket == ""

    def test_gcs_prefix_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.gcs_prefix == "rag_data/"

    def test_vertex_rag_corpus_id_default_is_empty(self) -> None:
        cfg = ExportConfig()
        assert cfg.vertex_rag_corpus_id == ""


class TestExportConfigFromDict:
    """Verify ExportConfig can be instantiated from a dict (YAML-loaded values)."""

    def test_full_config_from_dict(self) -> None:
        data = {
            "enabled": True,
            "gcp_project_id": "my-gcp-project",
            "bigquery_dataset": "my_dataset",
            "auto_export_reports": False,
            "auto_export_telemetry": True,
            "gcs_bucket": "my-bucket",
            "gcs_prefix": "exports/vaig/",
            "vertex_rag_corpus_id": "projects/my-gcp-project/locations/us-central1/ragCorpora/123",
        }
        cfg = ExportConfig(**data)

        assert cfg.enabled is True
        assert cfg.bigquery_dataset == "my_dataset"
        assert cfg.gcp_project_id == "my-gcp-project"
        assert cfg.auto_export_reports is False
        assert cfg.auto_export_telemetry is True
        assert cfg.gcs_bucket == "my-bucket"
        assert cfg.gcs_prefix == "exports/vaig/"
        assert cfg.vertex_rag_corpus_id == "projects/my-gcp-project/locations/us-central1/ragCorpora/123"

    def test_legacy_aliases_still_parse(self) -> None:
        cfg = ExportConfig(
            bigquery_project="legacy-project",
            rag_corpus_name="legacy-corpus",
        )

        assert cfg.gcp_project_id == "legacy-project"
        assert cfg.bigquery_project == "legacy-project"
        assert cfg.vertex_rag_corpus_id == "legacy-corpus"
        assert cfg.rag_corpus_name == "legacy-corpus"

    def test_partial_dict_uses_defaults(self) -> None:
        """Only override some fields — rest should use defaults."""
        cfg = ExportConfig(**{"enabled": True, "gcs_bucket": "my-bucket"})

        assert cfg.enabled is True
        assert cfg.gcs_bucket == "my-bucket"
        # These retain defaults
        assert cfg.bigquery_dataset == "vaig_analytics"
        assert cfg.gcs_prefix == "rag_data/"
        assert cfg.gcp_project_id == ""


class TestExportConfigIntegration:
    """Verify ExportConfig is integrated into the Settings class."""

    def test_settings_has_export_field(self) -> None:
        settings = Settings()
        assert hasattr(settings, "export")
        assert isinstance(settings.export, ExportConfig)

    def test_settings_export_defaults(self) -> None:
        """Settings.export must default to ExportConfig() — export is off by default."""
        settings = Settings()
        assert settings.export.enabled is False

    def test_settings_export_from_dict(self) -> None:
        """Simulate what Settings.load() does when it finds an export: section in YAML."""
        settings = Settings(
            **{"export": {"enabled": True, "bigquery_dataset": "prod_analytics"}}
        )
        assert settings.export.enabled is True
        assert settings.export.bigquery_dataset == "prod_analytics"
        # Non-overridden fields keep defaults
        assert settings.export.gcs_prefix == "rag_data/"

    def test_export_disabled_does_not_affect_other_settings(self) -> None:
        """Ensure ExportConfig fields don't bleed into unrelated settings."""
        settings = Settings()
        assert settings.telemetry.enabled is True  # unaffected
        assert settings.export.enabled is False

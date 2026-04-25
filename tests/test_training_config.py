"""Tests for TrainingConfig Pydantic model."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.config import TrainingConfig


class TestTrainingConfigDefaults:
    """Verify all default values match the spec (REQ-TRAIN-01)."""

    def test_enabled_is_false_by_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.enabled is False

    def test_base_model_default_is_sentinel(self) -> None:
        """base_model defaults to '' sentinel; Settings._resolve_model_sentinels fills it."""
        cfg = TrainingConfig()
        assert cfg.base_model == ""

    def test_base_model_resolved_via_settings(self) -> None:
        """Settings resolves the sentinel to the fallback model at construction time."""
        from vaig.core.config import Settings

        settings = Settings()
        assert settings.training.base_model != ""
        assert settings.training.base_model == settings.models.fallback

    def test_min_examples_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.min_examples == 50

    def test_max_examples_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.max_examples == 10000

    def test_min_rating_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.min_rating == 4

    def test_output_dir_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.output_dir == Path("training_data")

    def test_epochs_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.epochs == 3

    def test_learning_rate_multiplier_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.learning_rate_multiplier == 1.0

    def test_gcs_staging_prefix_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.gcs_staging_prefix == "training_data/"


class TestTrainingConfigValidation:
    """Verify Pydantic validators enforce bounds (REQ-TRAIN-01)."""

    def test_min_examples_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="min_examples must be >= 10"):
            TrainingConfig(min_examples=3)

    def test_min_examples_at_boundary_10_is_valid(self) -> None:
        cfg = TrainingConfig(min_examples=10)
        assert cfg.min_examples == 10

    def test_max_examples_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match="max_examples must be <= 100000"):
            TrainingConfig(max_examples=100001)

    def test_max_examples_at_boundary_100000_is_valid(self) -> None:
        cfg = TrainingConfig(max_examples=100000)
        assert cfg.max_examples == 100000

    def test_min_rating_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match="min_rating must be between 1 and 5"):
            TrainingConfig(min_rating=0)

    def test_min_rating_above_5_raises(self) -> None:
        with pytest.raises(ValueError, match="min_rating must be between 1 and 5"):
            TrainingConfig(min_rating=6)

    def test_min_rating_boundary_values_valid(self) -> None:
        cfg1 = TrainingConfig(min_rating=1)
        assert cfg1.min_rating == 1
        cfg5 = TrainingConfig(min_rating=5)
        assert cfg5.min_rating == 5

    def test_epochs_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match="epochs must be between 1 and 10"):
            TrainingConfig(epochs=0)

    def test_epochs_above_10_raises(self) -> None:
        with pytest.raises(ValueError, match="epochs must be between 1 and 10"):
            TrainingConfig(epochs=11)

    def test_epochs_boundary_values_valid(self) -> None:
        cfg1 = TrainingConfig(epochs=1)
        assert cfg1.epochs == 1
        cfg10 = TrainingConfig(epochs=10)
        assert cfg10.epochs == 10

    def test_learning_rate_multiplier_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate_multiplier must be > 0"):
            TrainingConfig(learning_rate_multiplier=0)

    def test_learning_rate_multiplier_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate_multiplier must be > 0"):
            TrainingConfig(learning_rate_multiplier=-0.5)

    def test_learning_rate_multiplier_positive_valid(self) -> None:
        cfg = TrainingConfig(learning_rate_multiplier=0.1)
        assert cfg.learning_rate_multiplier == 0.1

    def test_gcs_staging_prefix_normalized(self) -> None:
        cfg = TrainingConfig(gcs_staging_prefix="my_prefix")
        assert cfg.gcs_staging_prefix == "my_prefix/"

    def test_gcs_staging_prefix_trailing_slash_preserved(self) -> None:
        cfg = TrainingConfig(gcs_staging_prefix="my_prefix/")
        assert cfg.gcs_staging_prefix == "my_prefix/"


class TestTrainingConfigFromDict:
    """Verify TrainingConfig can be instantiated from a dict (YAML-loaded values)."""

    def test_full_config_from_dict(self) -> None:
        data = {
            "enabled": True,
            "base_model": "gemini-2.0-flash-lite-001",
            "min_examples": 100,
            "max_examples": 5000,
            "min_rating": 3,
            "output_dir": "/tmp/train",
            "epochs": 5,
            "learning_rate_multiplier": 0.5,
            "gcs_staging_prefix": "custom_prefix/",
        }
        cfg = TrainingConfig(**data)
        assert cfg.enabled is True
        assert cfg.base_model == "gemini-2.0-flash-lite-001"
        assert cfg.min_examples == 100
        assert cfg.max_examples == 5000
        assert cfg.min_rating == 3
        assert cfg.output_dir == Path("/tmp/train")
        assert cfg.epochs == 5
        assert cfg.learning_rate_multiplier == 0.5
        assert cfg.gcs_staging_prefix == "custom_prefix/"

"""Tests for model sentinel resolution in Settings (DMD-01).

Verifies that empty-string sentinel fields in ModelsConfig, AgentsConfig,
and TrainingConfig are filled with derived defaults by the
``Settings._resolve_model_sentinels`` model_validator.
"""

from __future__ import annotations

from vaig.core.config import AgentsConfig, ModelsConfig, Settings, TrainingConfig


class TestSettingsModelSentinels:
    """Settings._resolve_model_sentinels fills empty fields from models.*."""

    def test_orchestrator_model_inherits_default(self) -> None:
        """orchestrator_model='' resolves to models.default."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            agents=AgentsConfig(orchestrator_model=""),
        )
        assert s.agents.orchestrator_model == "gemini-test-pro"

    def test_specialist_model_inherits_fallback(self) -> None:
        """specialist_model='' resolves to models.fallback."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            agents=AgentsConfig(specialist_model=""),
        )
        assert s.agents.specialist_model == "gemini-test-flash"

    def test_explicit_orchestrator_model_not_overwritten(self) -> None:
        """orchestrator_model='custom-model' stays unchanged."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            agents=AgentsConfig(orchestrator_model="custom-model"),
        )
        assert s.agents.orchestrator_model == "custom-model"

    def test_explicit_specialist_model_not_overwritten(self) -> None:
        """specialist_model='custom-model' stays unchanged."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            agents=AgentsConfig(specialist_model="custom-model"),
        )
        assert s.agents.specialist_model == "custom-model"

    def test_training_base_model_inherits_fallback(self) -> None:
        """training.base_model='' resolves to models.fallback."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            training=TrainingConfig(base_model=""),
        )
        assert s.training.base_model == "gemini-test-flash"

    def test_training_base_model_explicit_not_overwritten(self) -> None:
        """training.base_model='tuning-model' stays unchanged."""
        s = Settings(
            models=ModelsConfig(default="gemini-test-pro", fallback="gemini-test-flash"),
            training=TrainingConfig(base_model="tuning-model"),
        )
        assert s.training.base_model == "tuning-model"

    def test_models_default_sentinel_gets_hardcoded_fallback(self) -> None:
        """models.default='' resolves to hard-coded last-resort 'gemini-2.5-pro'."""
        s = Settings(models=ModelsConfig(default="", fallback="gemini-test-flash"))
        assert s.models.default == "gemini-2.5-pro"

    def test_models_fallback_sentinel_gets_hardcoded_fallback(self) -> None:
        """models.fallback='' resolves to hard-coded last-resort 'gemini-2.5-flash'."""
        s = Settings(models=ModelsConfig(default="gemini-test-pro", fallback=""))
        assert s.models.fallback == "gemini-2.5-flash"

    def test_both_sentinels_resolved_bare_construction(self) -> None:
        """Bare Settings() always produces non-empty model strings."""
        s = Settings()
        assert s.models.default
        assert s.models.fallback
        assert s.agents.orchestrator_model
        assert s.agents.specialist_model
        assert s.training.base_model


class TestAgentConfigEffectiveModel:
    """AgentConfig.effective_model() returns model or falls back to default."""

    def test_returns_own_model_when_set(self) -> None:
        from vaig.agents.base import AgentConfig

        cfg = AgentConfig(
            name="test",
            role="tester",
            system_instruction="do stuff",
            model="gemini-custom",
        )
        assert cfg.effective_model("gemini-default") == "gemini-custom"

    def test_returns_default_when_empty(self) -> None:
        from vaig.agents.base import AgentConfig

        cfg = AgentConfig(
            name="test",
            role="tester",
            system_instruction="do stuff",
            model="",
        )
        assert cfg.effective_model("gemini-default") == "gemini-default"

    def test_returns_default_when_not_supplied(self) -> None:
        from vaig.agents.base import AgentConfig

        cfg = AgentConfig(
            name="test",
            role="tester",
            system_instruction="do stuff",
        )
        assert cfg.effective_model("gemini-fallback") == "gemini-fallback"

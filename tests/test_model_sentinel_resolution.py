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


class TestSupportsThinking:
    """supports_thinking() uses is-not-None check and unions extra_prefixes."""

    def test_builtin_model_detected_without_extra_prefixes(self) -> None:
        from vaig.core.config import supports_thinking

        assert supports_thinking("gemini-2.5-pro") is True
        assert supports_thinking("gemini-2.5-flash-001") is True

    def test_unknown_model_returns_false(self) -> None:
        from vaig.core.config import supports_thinking

        assert supports_thinking("some-other-model") is False

    def test_empty_extra_prefixes_does_not_disable_builtins(self) -> None:
        """extra_prefixes=[] → union with THINKING_CAPABLE_MODELS, builtins still work."""
        from vaig.core.config import supports_thinking

        assert supports_thinking("gemini-2.5-pro", extra_prefixes=[]) is True

    def test_custom_prefix_added_via_extra_prefixes(self) -> None:
        """Extra prefix extends detection without removing builtins."""
        from vaig.core.config import supports_thinking

        assert supports_thinking("my-custom-thinking-model", extra_prefixes=["my-custom"]) is True
        assert supports_thinking("gemini-2.5-pro", extra_prefixes=["my-custom"]) is True

    def test_none_extra_prefixes_uses_only_builtins(self) -> None:
        """extra_prefixes=None → only THINKING_CAPABLE_MODELS consulted."""
        from vaig.core.config import supports_thinking

        assert supports_thinking("gemini-3-flash", extra_prefixes=None) is True
        assert supports_thinking("unknown-model", extra_prefixes=None) is False

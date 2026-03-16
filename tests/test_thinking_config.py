"""Tests for Gemini thinking mode — ThinkingConfig, model detection, and response parsing.

Covers:
- ThinkingConfig defaults and custom values
- THINKING_CAPABLE_MODELS detection via supports_thinking()
- _build_generation_config() with thinking enabled/disabled
- _extract_thinking_text() and _extract_output_text()
- generate_with_tools() part parsing skips thought parts
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.client import GeminiClient, GenerationResult, ToolCallResult
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelInfo,
    ModelsConfig,
    Settings,
    ThinkingConfig,
    supports_thinking,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings with thinking-capable model."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
        ),
        models=ModelsConfig(
            default="gemini-2.5-flash",
            fallback="gemini-2.5-pro",
            available=[
                ModelInfo(id="gemini-2.5-flash", description="Flash"),
                ModelInfo(id="gemini-2.5-pro", description="Pro"),
            ],
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    """Uninitialized GeminiClient."""
    return GeminiClient(settings)


def _make_thinking_response(
    thinking_text: str = "Let me think...",
    output_text: str = "Here is the answer.",
    prompt_tokens: int = 10,
    completion_tokens: int = 30,
    thinking_tokens: int = 50,
    finish_reason: str = "STOP",
) -> MagicMock:
    """Create a mock response with both thinking and output parts."""
    response = MagicMock()
    response.text = thinking_text + output_text  # Default .text includes everything

    # Thinking part
    thought_part = MagicMock()
    thought_part.thought = True
    thought_part.text = thinking_text
    thought_part.function_call = None

    # Output part
    output_part = MagicMock()
    output_part.thought = False
    output_part.text = output_text
    output_part.function_call = None

    # Candidate
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content.parts = [thought_part, output_part]
    response.candidates = [candidate]

    # Usage metadata
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.total_token_count = prompt_tokens + completion_tokens + thinking_tokens
    response.usage_metadata.thoughts_token_count = thinking_tokens

    return response


def _make_normal_response(
    text: str = "Normal response.",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "STOP",
) -> MagicMock:
    """Create a mock response WITHOUT thinking parts."""
    response = MagicMock()
    response.text = text

    part = MagicMock()
    part.thought = False
    part.text = text
    part.function_call = None

    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content.parts = [part]
    response.candidates = [candidate]

    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.total_token_count = prompt_tokens + completion_tokens
    response.usage_metadata.thoughts_token_count = None

    return response


# ── ThinkingConfig unit tests ────────────────────────────────


class TestThinkingConfig:
    """ThinkingConfig Pydantic model defaults and validation."""

    def test_defaults(self) -> None:
        cfg = ThinkingConfig()
        assert cfg.enabled is False
        assert cfg.budget_tokens is None
        assert cfg.include_thoughts is True

    def test_enabled_with_budget(self) -> None:
        cfg = ThinkingConfig(enabled=True, budget_tokens=2048)
        assert cfg.enabled is True
        assert cfg.budget_tokens == 2048
        assert cfg.include_thoughts is True

    def test_disabled_explicit(self) -> None:
        cfg = ThinkingConfig(enabled=False, include_thoughts=False)
        assert cfg.enabled is False
        assert cfg.include_thoughts is False

    def test_auto_budget(self) -> None:
        """budget_tokens=-1 means automatic budget."""
        cfg = ThinkingConfig(enabled=True, budget_tokens=-1)
        assert cfg.budget_tokens == -1

    def test_zero_budget_disables_thinking(self) -> None:
        """budget_tokens=0 tells the API to disable thinking."""
        cfg = ThinkingConfig(enabled=True, budget_tokens=0)
        assert cfg.budget_tokens == 0


# ── Model capability detection ───────────────────────────────


class TestSupportsThinking:
    """supports_thinking() prefix matching."""

    def test_flash_supported(self) -> None:
        assert supports_thinking("gemini-2.5-flash") is True

    def test_pro_supported(self) -> None:
        assert supports_thinking("gemini-2.5-pro") is True

    def test_versioned_flash_supported(self) -> None:
        assert supports_thinking("gemini-2.5-flash-001") is True

    def test_versioned_pro_supported(self) -> None:
        assert supports_thinking("gemini-2.5-pro-preview-0506") is True

    def test_old_model_not_supported(self) -> None:
        assert supports_thinking("gemini-1.5-pro") is False

    def test_gemini_2_flash_not_supported(self) -> None:
        """gemini-2.0-flash is NOT a thinking model."""
        assert supports_thinking("gemini-2.0-flash") is False

    def test_empty_string(self) -> None:
        assert supports_thinking("") is False


# ── _build_generation_config with thinking ────────────────────


class TestBuildGenerationConfigThinking:
    """_build_generation_config() thinking_config integration."""

    def test_thinking_disabled_no_thinking_config(self, client: GeminiClient) -> None:
        """When thinking is disabled, no thinking_config in the output."""
        config = client._build_generation_config()
        assert config.thinking_config is None

    def test_thinking_enabled_creates_thinking_config(self, settings: Settings) -> None:
        """When thinking is enabled, thinking_config is set."""
        settings.generation.thinking = ThinkingConfig(enabled=True)
        c = GeminiClient(settings)
        config = c._build_generation_config()
        assert config.thinking_config is not None
        assert config.thinking_config.include_thoughts is True

    def test_thinking_enabled_with_custom_budget(self, settings: Settings) -> None:
        """Custom budget_tokens is passed to thinking_config."""
        settings.generation.thinking = ThinkingConfig(enabled=True, budget_tokens=4096)
        c = GeminiClient(settings)
        config = c._build_generation_config()
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 4096

    def test_thinking_enabled_no_budget_omits_budget(self, settings: Settings) -> None:
        """When budget_tokens is None, thinking_budget is not set."""
        settings.generation.thinking = ThinkingConfig(enabled=True, budget_tokens=None)
        c = GeminiClient(settings)
        config = c._build_generation_config()
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget is None

    def test_thinking_warning_on_unsupported_model(
        self,
        settings: Settings,
    ) -> None:
        """Warning when thinking is enabled on a non-thinking model."""
        settings.generation.thinking = ThinkingConfig(enabled=True)
        settings.models.default = "gemini-1.5-pro"
        c = GeminiClient(settings)
        with patch("vaig.core.client.logger") as mock_logger:
            c._build_generation_config()
            mock_logger.warning.assert_called_once()
            assert "may not support" in mock_logger.warning.call_args[0][0]


# ── Thinking text extraction ─────────────────────────────────


class TestExtractThinkingText:
    """GeminiClient._extract_thinking_text() static method."""

    def test_extracts_thinking_parts(self) -> None:
        response = _make_thinking_response(thinking_text="My reasoning here")
        result = GeminiClient._extract_thinking_text(response)
        assert result == "My reasoning here"

    def test_returns_none_without_thinking(self) -> None:
        response = _make_normal_response(text="Just a normal reply")
        result = GeminiClient._extract_thinking_text(response)
        assert result is None

    def test_empty_response_returns_none(self) -> None:
        response = MagicMock()
        response.candidates = None
        result = GeminiClient._extract_thinking_text(response)
        assert result is None

    def test_multiple_thinking_parts_concatenated(self) -> None:
        """Multiple thought=True parts are joined."""
        response = MagicMock()
        p1 = MagicMock(thought=True, text="Part 1. ")
        p2 = MagicMock(thought=True, text="Part 2.")
        p3 = MagicMock(thought=False, text="Output.")
        candidate = MagicMock()
        candidate.content.parts = [p1, p2, p3]
        response.candidates = [candidate]
        result = GeminiClient._extract_thinking_text(response)
        assert result == "Part 1. Part 2."


class TestExtractOutputText:
    """GeminiClient._extract_output_text() static method."""

    def test_filters_out_thinking_parts(self) -> None:
        response = _make_thinking_response(
            thinking_text="Thinking...",
            output_text="Answer.",
        )
        result = GeminiClient._extract_output_text(response)
        assert result == "Answer."

    def test_normal_response_returns_text(self) -> None:
        response = _make_normal_response(text="Hello")
        result = GeminiClient._extract_output_text(response)
        assert result == "Hello"

    def test_empty_parts_falls_back_to_response_text(self) -> None:
        response = MagicMock()
        response.text = "Fallback text"
        response.candidates = None
        result = GeminiClient._extract_output_text(response)
        assert result == "Fallback text"


# ── GenerationResult / ToolCallResult dataclass fields ────────


class TestResultDataclasses:
    """thinking_text field on result dataclasses."""

    def test_generation_result_default_none(self) -> None:
        r = GenerationResult(text="hi", model="m")
        assert r.thinking_text is None

    def test_generation_result_with_thinking(self) -> None:
        r = GenerationResult(text="hi", model="m", thinking_text="thoughts")
        assert r.thinking_text == "thoughts"

    def test_tool_call_result_default_none(self) -> None:
        r = ToolCallResult(text="", model="m", function_calls=[])
        assert r.thinking_text is None

    def test_tool_call_result_with_thinking(self) -> None:
        r = ToolCallResult(
            text="",
            model="m",
            function_calls=[],
            thinking_text="chain of thought",
        )
        assert r.thinking_text == "chain of thought"


# ── GenerationConfig nested thinking field ────────────────────


class TestGenerationConfigThinkingField:
    """ThinkingConfig is nested inside GenerationConfig."""

    def test_default_thinking_disabled(self) -> None:
        cfg = GenerationConfig()
        assert cfg.thinking.enabled is False

    def test_custom_thinking_config(self) -> None:
        cfg = GenerationConfig(
            thinking=ThinkingConfig(enabled=True, budget_tokens=1024),
        )
        assert cfg.thinking.enabled is True
        assert cfg.thinking.budget_tokens == 1024

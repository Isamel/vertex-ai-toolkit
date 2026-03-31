"""Tests for GeminiClient — google-genai SDK wrapper with mocked responses."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, StreamResult
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelInfo,
    ModelsConfig,
    SafetyConfig,
    SafetySettingConfig,
    Settings,
)

# ── Fixtures ─────────────────────────────────────────────────
# _reset_settings is provided by conftest.py (autouse)


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings for client tests."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
        ),
        models=ModelsConfig(
            default="gemini-2.5-pro",
            fallback="gemini-2.5-flash",
            available=[
                ModelInfo(id="gemini-2.5-pro", description="Pro model"),
                ModelInfo(id="gemini-2.5-flash", description="Flash model"),
            ],
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    """An uninitialized GeminiClient."""
    return GeminiClient(settings)


def _make_mock_response(
    text: str = "Hello from Gemini",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    finish_reason: str = "STOP",
) -> MagicMock:
    """Create a mock google-genai GenerateContentResponse."""
    response = MagicMock()
    response.text = text

    # usage_metadata
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.total_token_count = total_tokens

    # candidates
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    response.candidates = [candidate]

    return response


def _make_mock_stream_chunks(
    texts: list[str],
    *,
    last_usage: dict[str, int] | None = None,
) -> list[MagicMock]:
    """Create a list of mock streaming chunks.

    By default, chunks have ``usage_metadata = None`` (no usage info).
    Pass *last_usage* to attach usage metadata to the **last** chunk, matching
    how the Gemini SDK reports cumulative totals on the final streamed chunk.
    """
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.text = t
        chunk.usage_metadata = None
        chunks.append(chunk)
    if last_usage and chunks:
        um = MagicMock()
        um.prompt_token_count = last_usage.get("prompt_tokens", 0)
        um.candidates_token_count = last_usage.get("completion_tokens", 0)
        um.total_token_count = last_usage.get("total_tokens", 0)
        um.thoughts_token_count = last_usage.get("thinking_tokens", 0)
        chunks[-1].usage_metadata = um
    return chunks


# ── TestGeminiClientInit ─────────────────────────────────────


class TestGeminiClientInit:
    """Tests for GeminiClient constructor and initialize()."""

    def test_constructor_sets_defaults(self, settings: Settings) -> None:
        client = GeminiClient(settings)
        assert client._initialized is False
        assert client._current_model_id == "gemini-2.5-pro"
        assert client._client is None

    def test_current_model_property(self, client: GeminiClient) -> None:
        assert client.current_model == "gemini-2.5-pro"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_initialize_creates_genai_client(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        fake_creds = MagicMock()
        mock_get_creds.return_value = fake_creds

        client.initialize()

        mock_get_creds.assert_called_once_with(client._settings)
        mock_genai_client_cls.assert_called_once()
        call_kwargs = mock_genai_client_cls.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["location"] == "us-central1"
        assert call_kwargs["credentials"] is fake_creds
        assert "http_options" in call_kwargs  # SDK-level retry options
        assert client._initialized is True
        assert client._client is mock_genai_client_cls.return_value

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_initialize_is_idempotent(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        client.initialize()
        client.initialize()

        assert mock_genai_client_cls.call_count == 1

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_ensure_initialized_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        client._ensure_initialized()

        assert client._initialized is True
        mock_genai_client_cls.assert_called_once()


# ── TestSwitchModel ──────────────────────────────────────────


class TestSwitchModel:
    """Tests for switch_model()."""

    def test_switch_model_returns_new_id(self, client: GeminiClient) -> None:
        result = client.switch_model("gemini-2.5-flash")
        assert result == "gemini-2.5-flash"

    def test_switch_model_updates_current(self, client: GeminiClient) -> None:
        client.switch_model("gemini-2.5-flash")
        assert client.current_model == "gemini-2.5-flash"

    def test_switch_model_to_same(self, client: GeminiClient) -> None:
        result = client.switch_model("gemini-2.5-pro")
        assert result == "gemini-2.5-pro"
        assert client.current_model == "gemini-2.5-pro"


# ── TestReinitialize ────────────────────────────────────────


class TestReinitialize:
    """Tests for reinitialize() method."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_with_new_project(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        client.reinitialize(project="new-project")

        mock_genai_client_cls.assert_called_once()
        call_kwargs = mock_genai_client_cls.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "new-project"
        assert call_kwargs["location"] == "us-central1"
        assert call_kwargs["credentials"] is mock_get_creds.return_value
        assert "http_options" in call_kwargs
        assert client._initialized is True
        assert client._settings.gcp.project_id == "new-project"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_with_new_location(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        client.reinitialize(location="europe-west1")

        mock_genai_client_cls.assert_called_once()
        call_kwargs = mock_genai_client_cls.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["location"] == "europe-west1"
        assert call_kwargs["credentials"] is mock_get_creds.return_value
        assert "http_options" in call_kwargs
        assert client._active_location == "europe-west1"
        assert client._settings.gcp.location == "europe-west1"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_with_both(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        client.reinitialize(project="proj-2", location="asia-east1")

        mock_genai_client_cls.assert_called_once()
        call_kwargs = mock_genai_client_cls.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "proj-2"
        assert call_kwargs["location"] == "asia-east1"
        assert call_kwargs["credentials"] is mock_get_creds.return_value
        assert "http_options" in call_kwargs

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_resets_fallback_flag(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        client._using_fallback = True

        client.reinitialize(project="new-project")

        assert client._using_fallback is False

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_replaces_client(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        old_client = MagicMock()
        client._client = old_client

        client.reinitialize(project="new-project")

        assert client._client is not old_client
        assert client._client is mock_genai_client_cls.return_value

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_reinitialize_failure_rolls_back(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_genai_client_cls.side_effect = Exception("auth failed")

        from vaig.core.exceptions import GeminiClientError

        with pytest.raises(GeminiClientError, match="auth failed"):
            client.reinitialize(project="bad-project", location="bad-loc")

        # Settings must be rolled back
        assert client._settings.gcp.project_id == "test-project"
        assert client._active_location == "us-central1"
        assert client._settings.gcp.location == "us-central1"


# ── TestBuildGenerationConfig ────────────────────────────────


class TestBuildGenerationConfig:
    """Tests for _build_generation_config().

    The new SDK uses types.GenerateContentConfig. We patch it
    at the module level where it's used.
    """

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_uses_settings_defaults(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        client._build_generation_config()

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_overrides_take_precedence(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        client._build_generation_config(temperature=0.0, max_output_tokens=1024)

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.0,
            max_output_tokens=1024,
            top_p=0.95,
            top_k=40,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_partial_overrides(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        client._build_generation_config(top_k=10)

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=10,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_frequency_penalty_included_when_set(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """frequency_penalty is only sent when explicitly provided."""
        client._build_generation_config(frequency_penalty=0.5)

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
            frequency_penalty=0.5,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_presence_penalty_included_when_set(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """presence_penalty is only sent when explicitly provided."""
        client._build_generation_config(presence_penalty=0.3)

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
            presence_penalty=0.3,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_both_penalties_included_together(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Both penalty params included when both provided."""
        client._build_generation_config(
            frequency_penalty=0.5, presence_penalty=0.2,
        )

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
            frequency_penalty=0.5,
            presence_penalty=0.2,
        )

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_penalties_omitted_by_default(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Without explicit penalty params, they are NOT included in kwargs."""
        client._build_generation_config()

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_system_instruction_included_when_set(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """system_instruction is now part of GenerateContentConfig in the new SDK."""
        client._build_generation_config(system_instruction="Be concise.")

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["system_instruction"] == "Be concise."

    @patch("vaig.core.client.types.SafetySetting")
    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_safety_settings_included_when_configured(
        self,
        mock_gen_config_cls: MagicMock,
        mock_safety_setting_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """safety_settings are passed to GenerateContentConfig when enabled + non-empty."""
        settings.safety = SafetyConfig(
            enabled=True,
            settings=[
                SafetySettingConfig(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                SafetySettingConfig(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
            ],
        )
        client = GeminiClient(settings)

        mock_safety_setting_cls.side_effect = lambda **kw: kw

        client._build_generation_config()

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert "safety_settings" in call_kwargs
        assert len(call_kwargs["safety_settings"]) == 2
        assert call_kwargs["safety_settings"][0] == {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH",
        }
        assert call_kwargs["safety_settings"][1] == {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        }

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_safety_settings_omitted_when_disabled(
        self,
        mock_gen_config_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """safety_settings NOT passed when safety.enabled is False."""
        settings.safety = SafetyConfig(
            enabled=False,
            settings=[
                SafetySettingConfig(category="HARM_CATEGORY_HARASSMENT"),
            ],
        )
        client = GeminiClient(settings)

        client._build_generation_config()

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert "safety_settings" not in call_kwargs

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_safety_settings_omitted_when_empty(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """safety_settings NOT passed when settings list is empty (default)."""
        client._build_generation_config()

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert "safety_settings" not in call_kwargs

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_response_schema_included_when_set(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """response_schema is only sent when explicitly provided."""
        from pydantic import BaseModel

        class MyReport(BaseModel):
            title: str

        client._build_generation_config(response_schema=MyReport)

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["response_schema"] is MyReport

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_response_mime_type_included_when_set(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """response_mime_type is only sent when explicitly provided."""
        client._build_generation_config(response_mime_type="application/json")

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["response_mime_type"] == "application/json"

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_both_schema_params_included_together(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Both response_schema and response_mime_type included when both provided."""
        from pydantic import BaseModel

        class MyReport(BaseModel):
            title: str

        client._build_generation_config(
            response_schema=MyReport,
            response_mime_type="application/json",
        )

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["response_schema"] is MyReport
        assert call_kwargs["response_mime_type"] == "application/json"

    @patch("vaig.core.client.types.GenerateContentConfig")
    def test_schema_params_omitted_by_default(
        self,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Without explicit schema params, they are NOT included in kwargs."""
        client._build_generation_config()

        call_kwargs = mock_gen_config_cls.call_args[1]
        assert "response_schema" not in call_kwargs
        assert "response_mime_type" not in call_kwargs


# ── TestBuildHistory ─────────────────────────────────────────


class TestBuildHistory:
    """Tests for _build_history() static method."""

    @patch("vaig.core.client.types.Part.from_text")
    @patch("vaig.core.client.types.Content")
    def test_builds_from_text_messages(
        self,
        mock_content_cls: MagicMock,
        mock_from_text: MagicMock,
    ) -> None:
        mock_from_text.side_effect = lambda text: f"part:{text}"

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="model", content="Hi there!"),
        ]

        result = GeminiClient._build_history(messages)

        assert len(result) == 2
        mock_content_cls.assert_any_call(role="user", parts=["part:Hello"])
        mock_content_cls.assert_any_call(role="model", parts=["part:Hi there!"])

    @patch("vaig.core.client.types.Content")
    def test_uses_parts_when_available(
        self,
        mock_content_cls: MagicMock,
    ) -> None:
        custom_parts = [MagicMock(), MagicMock()]
        messages = [
            ChatMessage(role="user", content="ignored", parts=custom_parts),
        ]

        result = GeminiClient._build_history(messages)

        assert len(result) == 1
        mock_content_cls.assert_called_once_with(role="user", parts=custom_parts)

    def test_empty_history_returns_empty_list(self) -> None:
        result = GeminiClient._build_history([])
        assert result == []


# ── TestGeminiClientGenerate ─────────────────────────────────


class TestGeminiClientGenerate:
    """Tests for generate() — non-streaming generation."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_single_turn_generation(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(
            text="Generated text",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            finish_reason="STOP",
        )
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        result = client.generate("Tell me a joke")

        assert isinstance(result, GenerationResult)
        assert result.text == "Generated text"
        assert result.model == "gemini-2.5-pro"
        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "thinking_tokens": 0,
        }
        assert result.finish_reason == "STOP"
        mock_genai.models.generate_content.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_multi_turn_generation_with_history(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Follow-up answer")

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        history = [
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(role="model", content="A programming language."),
        ]

        result = client.generate("Tell me more", history=history)

        assert result.text == "Follow-up answer"
        mock_genai.chats.create.assert_called_once()
        mock_chat.send_message.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_with_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Flash response")
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        result = client.generate("Hello", model_id="gemini-2.5-flash")

        assert result.model == "gemini-2.5-flash"
        # Verify the model ID was passed to generate_content
        call_kwargs = mock_genai.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_with_system_instruction(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Concise answer")
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        result = client.generate("Explain AI", system_instruction="Be concise.")

        assert result.text == "Concise answer"
        # System instruction is now part of the config, verified via generate_content call
        mock_genai.models.generate_content.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_without_usage_metadata(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "No usage info"
        mock_response.usage_metadata = None
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        mock_response.candidates = [candidate]

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        result = client.generate("Hello")

        assert result.usage == {}

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_with_no_candidates(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Empty candidates")
        mock_response.candidates = []

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        result = client.generate("Hello")

        assert result.finish_reason == ""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate() should auto-init if not yet initialized."""
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response()
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = mock_response

        assert client._initialized is False
        client.generate("Hello")
        assert client._initialized is True
        mock_genai_client_cls.assert_called_once()


# ── TestGeminiClientStream ───────────────────────────────────


class TestGeminiClientStream:
    """Tests for generate_stream() — streaming generation."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_stream_yields_text_chunks(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["Hello ", "world", "!"])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content_stream.return_value = chunks

        result = list(client.generate_stream("Say hello"))

        assert result == ["Hello ", "world", "!"]
        mock_genai.models.generate_content_stream.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_stream_skips_empty_chunks(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["Hello ", "", "world"])
        chunks[1].text = ""
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content_stream.return_value = chunks

        result = list(client.generate_stream("Say hello"))

        assert result == ["Hello ", "world"]

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_stream_with_history(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["More ", "info"])
        mock_chat = MagicMock()
        mock_chat.send_message_stream.return_value = chunks

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        history = [
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(role="model", content="A language."),
        ]

        result = list(client.generate_stream("Tell me more", history=history))

        assert result == ["More ", "info"]
        mock_genai.chats.create.assert_called_once()
        mock_chat.send_message_stream.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_stream_with_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["Fast!"])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content_stream.return_value = chunks

        result = list(client.generate_stream("Hello", model_id="gemini-2.5-flash"))

        assert result == ["Fast!"]
        call_kwargs = mock_genai.models.generate_content_stream.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_stream_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["Hi"])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content_stream.return_value = chunks

        assert client._initialized is False
        list(client.generate_stream("Hello"))
        assert client._initialized is True


# ── TestStreamResult ─────────────────────────────────────────


class TestStreamResult:
    """Tests for StreamResult — iterable wrapper with usage capture."""

    def test_iteration_yields_text_chunks(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello ", "world", "!"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        collected = list(result)

        assert collected == ["Hello ", "world", "!"]

    def test_skips_empty_text_chunks(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello ", "", "world"])
        chunks[1].text = ""  # Explicitly empty
        result = StreamResult(chunks, model="gemini-2.5-pro")

        collected = list(result)

        assert collected == ["Hello ", "world"]

    def test_text_property_accumulates_all_chunks(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello ", "world", "!"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        list(result)  # exhaust iterator

        assert result.text == "Hello world!"

    def test_usage_empty_when_no_usage_metadata(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        list(result)

        assert result.usage == {}

    def test_usage_captured_from_last_chunk(self) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 25,
            "total_tokens": 35,
            "thinking_tokens": 5,
        }
        chunks = _make_mock_stream_chunks(["Hello ", "world"], last_usage=usage)
        result = StreamResult(chunks, model="gemini-2.5-pro")

        list(result)

        assert result.usage == usage

    def test_usage_from_intermediate_chunk_overwritten_by_last(self) -> None:
        """If multiple chunks have usage, the last one wins."""
        chunks = _make_mock_stream_chunks(["a", "b"], last_usage={
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "thinking_tokens": 0,
        })
        # Also set usage on first chunk (simulating intermediate usage)
        um_early = MagicMock()
        um_early.prompt_token_count = 1
        um_early.candidates_token_count = 1
        um_early.total_token_count = 2
        um_early.thoughts_token_count = 0
        chunks[0].usage_metadata = um_early

        result = StreamResult(chunks, model="gemini-2.5-pro")
        list(result)

        # Last chunk wins
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 200

    def test_model_property(self) -> None:
        chunks = _make_mock_stream_chunks(["Hi"])
        result = StreamResult(chunks, model="gemini-2.5-flash")

        assert result.model == "gemini-2.5-flash"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_stream_returns_stream_result(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate_stream() now returns a StreamResult, not a bare generator."""
        mock_get_creds.return_value = MagicMock()
        usage = {
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "total_tokens": 45,
            "thinking_tokens": 0,
        }
        chunks = _make_mock_stream_chunks(["Hello ", "world"], last_usage=usage)
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content_stream.return_value = chunks

        stream_result = client.generate_stream("Say hello")

        assert isinstance(stream_result, StreamResult)

        # Backward compat: for-loop still works
        collected = list(stream_result)
        assert collected == ["Hello ", "world"]

        # Usage is now captured
        assert stream_result.usage == usage
        assert stream_result.text == "Hello world"
        assert stream_result.model == "gemini-2.5-pro"

    def test_can_iterate_with_for_loop(self) -> None:
        """Verify the most common calling pattern works."""
        chunks = _make_mock_stream_chunks(["one", "two", "three"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        parts = []
        for chunk in result:
            parts.append(chunk)

        assert parts == ["one", "two", "three"]


# ── TestCountTokens ──────────────────────────────────────────


class TestCountTokens:
    """Tests for count_tokens()."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_count_tokens_returns_total(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 42

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.count_tokens.return_value = mock_token_response

        count = client.count_tokens("How many tokens?")

        assert count == 42
        mock_genai.models.count_tokens.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_count_tokens_with_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 10
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.count_tokens.return_value = mock_token_response

        count = client.count_tokens("Hello", model_id="gemini-2.5-flash")

        assert count == 10
        call_kwargs = mock_genai.models.count_tokens.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_count_tokens_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 1
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.count_tokens.return_value = mock_token_response

        assert client._initialized is False
        client.count_tokens("hi")
        assert client._initialized is True


# ── TestListAvailableModels ──────────────────────────────────


class TestListAvailableModels:
    """Tests for list_available_models()."""

    def test_returns_configured_models(self, client: GeminiClient) -> None:
        models = client.list_available_models()

        assert len(models) == 2
        assert models[0] == {"id": "gemini-2.5-pro", "description": "Pro model"}
        assert models[1] == {"id": "gemini-2.5-flash", "description": "Flash model"}

    def test_returns_empty_when_no_models(self, settings: Settings) -> None:
        settings.models.available = []
        client = GeminiClient(settings)

        assert client.list_available_models() == []


# ── TestExtractUsage ─────────────────────────────────────────


class TestExtractUsage:
    """Tests for _extract_usage() static method."""

    def test_extracts_standard_fields(self) -> None:
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.thoughts_token_count = 0

        usage = GeminiClient._extract_usage(response)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["thinking_tokens"] == 0

    def test_extracts_thinking_tokens(self) -> None:
        """_extract_usage should capture thoughts_token_count."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 500
        response.usage_metadata.candidates_token_count = 200
        response.usage_metadata.total_token_count = 3700
        response.usage_metadata.thoughts_token_count = 3000

        usage = GeminiClient._extract_usage(response)

        assert usage["thinking_tokens"] == 3000

    def test_no_usage_metadata_returns_empty_dict(self) -> None:
        response = MagicMock()
        response.usage_metadata = None

        usage = GeminiClient._extract_usage(response)

        assert usage == {}

    def test_none_token_counts_default_to_zero(self) -> None:
        """Fields can be None (e.g. function-call-only responses)."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None
        response.usage_metadata.total_token_count = None
        # thoughts_token_count might not even exist as an attribute
        del response.usage_metadata.thoughts_token_count

        usage = GeminiClient._extract_usage(response)

        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["thinking_tokens"] == 0

    def test_missing_thoughts_attribute_defaults_to_zero(self) -> None:
        """When thoughts_token_count attribute doesn't exist, should be 0."""
        response = MagicMock(spec=["usage_metadata"])
        um = MagicMock(spec=["prompt_token_count", "candidates_token_count", "total_token_count"])
        um.prompt_token_count = 10
        um.candidates_token_count = 20
        um.total_token_count = 30
        response.usage_metadata = um

        usage = GeminiClient._extract_usage(response)

        assert usage["thinking_tokens"] == 0


# ── TestDataclasses ──────────────────────────────────────────


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_basic_construction(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.parts == []

    def test_with_parts(self) -> None:
        parts = [MagicMock(), MagicMock()]
        msg = ChatMessage(role="model", content="", parts=parts)
        assert len(msg.parts) == 2

    def test_default_parts_not_shared(self) -> None:
        msg1 = ChatMessage(role="user", content="a")
        msg2 = ChatMessage(role="user", content="b")
        msg1.parts.append("x")
        assert msg2.parts == []


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_basic_construction(self) -> None:
        result = GenerationResult(text="Hello", model="gemini-2.5-pro")
        assert result.text == "Hello"
        assert result.model == "gemini-2.5-pro"
        assert result.usage == {}
        assert result.finish_reason == ""

    def test_with_all_fields(self) -> None:
        result = GenerationResult(
            text="Response",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "total_tokens": 30},
            finish_reason="STOP",
        )
        assert result.usage["prompt_tokens"] == 10
        assert result.finish_reason == "STOP"

    def test_default_usage_not_shared(self) -> None:
        r1 = GenerationResult(text="a", model="m")
        r2 = GenerationResult(text="b", model="m")
        r1.usage["x"] = 1
        assert r2.usage == {}


# ── Async API Tests ─────────────────────────────────────────


class TestAsyncInitialize:
    """Tests for async_initialize()."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_initialize_creates_client(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        fake_creds = MagicMock()
        mock_get_creds.return_value = fake_creds

        await client.async_initialize()

        mock_get_creds.assert_called_once_with(client._settings)
        mock_genai_client_cls.assert_called_once()
        call_kwargs = mock_genai_client_cls.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["location"] == "us-central1"
        assert call_kwargs["credentials"] is fake_creds
        assert "http_options" in call_kwargs
        assert client._initialized is True

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_initialize_is_idempotent(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        await client.async_initialize()
        await client.async_initialize()

        assert mock_genai_client_cls.call_count == 1


class TestAsyncGenerate:
    """Tests for async_generate() — async non-streaming generation."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_single_turn_generation(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(
            text="Async generated text",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            finish_reason="STOP",
        )
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await client.async_generate("Tell me a joke")

        assert isinstance(result, GenerationResult)
        assert result.text == "Async generated text"
        assert result.model == "gemini-2.5-pro"
        assert result.usage["prompt_tokens"] == 5
        assert result.finish_reason == "STOP"
        mock_genai.aio.models.generate_content.assert_awaited_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_generate_with_history(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Async follow-up")

        mock_chat = MagicMock()
        mock_chat.send_message = AsyncMock(return_value=mock_response)

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.chats.create.return_value = mock_chat

        history = [
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(role="model", content="A programming language."),
        ]

        result = await client.async_generate("Tell me more", history=history)

        assert result.text == "Async follow-up"
        mock_genai.aio.chats.create.assert_called_once()
        mock_chat.send_message.assert_awaited_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_generate_with_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response(text="Flash async response")
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await client.async_generate("Hello", model_id="gemini-2.5-flash")

        assert result.model == "gemini-2.5-flash"

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_generate_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_response = _make_mock_response()
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        assert client._initialized is False
        await client.async_generate("Hello")
        assert client._initialized is True


class TestAsyncGenerateStream:
    """Tests for async_generate_stream() — async streaming generation."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_stream_yields_text_chunks(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["Hello ", "world", "!"])

        # The SDK's async generate_content_stream is a coroutine that returns
        # an async iterator — we simulate it by returning an async generator.
        async def _fake_stream(**kw: Any) -> Any:
            for c in chunks:
                yield c

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.generate_content_stream = AsyncMock(
            return_value=_fake_stream(),
        )

        stream_result = await client.async_generate_stream("Say hello")

        assert isinstance(stream_result, StreamResult)
        collected = list(stream_result)  # sync iteration works on materialised chunks
        assert collected == ["Hello ", "world", "!"]

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_stream_with_history(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        chunks = _make_mock_stream_chunks(["More ", "info"])

        async def _fake_stream(*a: Any, **kw: Any) -> Any:
            for c in chunks:
                yield c

        mock_chat = MagicMock()
        mock_chat.send_message_stream = AsyncMock(return_value=_fake_stream())
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.chats.create.return_value = mock_chat

        history = [
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(role="model", content="A language."),
        ]

        stream_result = await client.async_generate_stream("Tell me more", history=history)

        collected = list(stream_result)
        assert collected == ["More ", "info"]
        mock_genai.aio.chats.create.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_stream_captures_usage(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        usage = {
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "total_tokens": 45,
            "thinking_tokens": 0,
        }
        chunks = _make_mock_stream_chunks(["Hello ", "world"], last_usage=usage)

        async def _fake_stream(**kw: Any) -> Any:
            for c in chunks:
                yield c

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.generate_content_stream = AsyncMock(
            return_value=_fake_stream(),
        )

        stream_result = await client.async_generate_stream("Say hello")
        list(stream_result)  # exhaust

        assert stream_result.usage == usage
        assert stream_result.text == "Hello world"


class TestAsyncCountTokens:
    """Tests for async_count_tokens()."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_count_tokens_returns_total(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 42

        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.count_tokens = AsyncMock(return_value=mock_token_response)

        count = await client.async_count_tokens("How many tokens?")

        assert count == 42
        mock_genai.aio.models.count_tokens.assert_awaited_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_count_tokens_with_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        mock_token_response = MagicMock()
        mock_token_response.total_tokens = 10
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.aio.models.count_tokens = AsyncMock(return_value=mock_token_response)

        count = await client.async_count_tokens("Hello", model_id="gemini-2.5-flash")

        assert count == 10


class TestStreamResultAsyncIteration:
    """Tests for StreamResult async iteration (``async for``)."""

    async def test_async_iteration_yields_text_chunks(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello ", "world", "!"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        collected = []
        async for text in result:
            collected.append(text)

        assert collected == ["Hello ", "world", "!"]

    async def test_async_iteration_skips_empty_chunks(self) -> None:
        chunks = _make_mock_stream_chunks(["Hello ", "", "world"])
        chunks[1].text = ""
        result = StreamResult(chunks, model="gemini-2.5-pro")

        collected = []
        async for text in result:
            collected.append(text)

        assert collected == ["Hello ", "world"]

    async def test_async_iteration_captures_usage(self) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 25,
            "total_tokens": 35,
            "thinking_tokens": 5,
        }
        chunks = _make_mock_stream_chunks(["Hello ", "world"], last_usage=usage)
        result = StreamResult(chunks, model="gemini-2.5-pro")

        collected = []
        async for text in result:
            collected.append(text)

        assert result.usage == usage
        assert result.text == "Hello world"

    async def test_async_iteration_text_property(self) -> None:
        chunks = _make_mock_stream_chunks(["a", "b", "c"])
        result = StreamResult(chunks, model="gemini-2.5-pro")

        async for _ in result:
            pass

        assert result.text == "abc"


class TestAsyncRetryWithBackoff:
    """Tests for _async_retry_with_backoff()."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_retry_succeeds_on_first_attempt(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        async def _fn() -> str:
            return "success"

        result = await client._async_retry_with_backoff(_fn)
        assert result == "success"

    @patch("vaig.core.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_retry_retries_on_retryable_error(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        mock_sleep: AsyncMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()
        call_count = 0

        async def _fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise google_exceptions.ServiceUnavailable("503")
            return "recovered"

        result = await client._async_retry_with_backoff(_fn)
        assert result == "recovered"
        assert call_count == 2
        mock_sleep.assert_awaited_once()  # asyncio.sleep, not time.sleep

    @patch("vaig.core.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_retry_exhaustion_raises(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        mock_sleep: AsyncMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        async def _fn() -> str:
            raise google_exceptions.ServiceUnavailable("503 always")

        from vaig.core.exceptions import GeminiConnectionError

        with pytest.raises(GeminiConnectionError, match="retries exhausted"):
            await client._async_retry_with_backoff(_fn)

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    async def test_async_retry_propagates_non_retryable(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        mock_get_creds.return_value = MagicMock()

        async def _fn() -> str:
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await client._async_retry_with_backoff(_fn)


# ══════════════════════════════════════════════════════════════
# Quota integration — GeminiClient + QuotaChecker
# ══════════════════════════════════════════════════════════════


class TestQuotaIntegration:
    """Tests that GeminiClient calls QuotaChecker before API calls."""

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_calls_check_and_consume(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """check_and_consume is called before the API call in generate()."""
        mock_get_creds.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_genai_client_cls.return_value = mock_sdk
        mock_sdk.models.generate_content.return_value = _make_mock_response()

        quota_checker = MagicMock()
        client = GeminiClient(settings, quota_checker=quota_checker)

        with patch("vaig.core.identity.resolve_identity", return_value=("alice", "alice@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="alice:alice@co.com"):
                client.generate("Hello")

        quota_checker.check_and_consume.assert_called_once()
        call_args = quota_checker.check_and_consume.call_args
        assert call_args[0][0] == "alice:alice@co.com"  # user_key
        assert call_args[0][1] > 0  # estimated_tokens > 0

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_propagates_quota_exceeded(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """QuotaExceededError propagates to the caller."""
        mock_get_creds.return_value = MagicMock()

        from vaig.core.exceptions import QuotaExceededError

        quota_checker = MagicMock()
        quota_checker.check_and_consume.side_effect = QuotaExceededError(
            dimension="requests_per_day", used=501, limit=500, user_key="alice:alice@co.com"
        )
        client = GeminiClient(settings, quota_checker=quota_checker)

        with patch("vaig.core.identity.resolve_identity", return_value=("alice", "alice@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="alice:alice@co.com"):
                with pytest.raises(QuotaExceededError, match="requests_per_day"):
                    client.generate("Hello")

    def test_no_check_when_quota_checker_is_none(self, settings: Settings) -> None:
        """No quota check when quota_checker is None (disabled)."""
        client = GeminiClient(settings)
        assert client._quota_checker is None
        # _check_quota should be a no-op — no error raised
        client._check_quota("test prompt")

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_stream_calls_quota_check(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """check_and_consume is called in generate_stream() too."""
        mock_get_creds.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_genai_client_cls.return_value = mock_sdk
        mock_sdk.models.generate_content_stream.return_value = iter(
            _make_mock_stream_chunks(["Hello"])
        )

        quota_checker = MagicMock()
        client = GeminiClient(settings, quota_checker=quota_checker)

        with patch("vaig.core.identity.resolve_identity", return_value=("bob", "bob@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="bob:bob@co.com"):
                client.generate_stream("Streaming prompt")

        quota_checker.check_and_consume.assert_called_once()

    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_generate_with_tools_calls_quota_check(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        settings: Settings,
    ) -> None:
        """check_and_consume is called in generate_with_tools() too."""
        mock_get_creds.return_value = MagicMock()
        mock_sdk = MagicMock()
        mock_genai_client_cls.return_value = mock_sdk
        mock_sdk.models.generate_content.return_value = _make_mock_response()

        quota_checker = MagicMock()
        client = GeminiClient(settings, quota_checker=quota_checker)

        with patch("vaig.core.identity.resolve_identity", return_value=("carol", "carol@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="carol:carol@co.com"):
                client.generate_with_tools("Tool prompt", tool_declarations=[])

        quota_checker.check_and_consume.assert_called_once()

    def test_check_quota_estimates_string_tokens(self, settings: Settings) -> None:
        """_check_quota estimates tokens from string prompt length."""
        quota_checker = MagicMock()
        client = GeminiClient(settings, quota_checker=quota_checker)

        # "Hello world" = 11 chars → 11 // 4 = 2 tokens
        with patch("vaig.core.identity.resolve_identity", return_value=("u", "u@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="u:u@co.com"):
                client._check_quota("Hello world")

        call_args = quota_checker.check_and_consume.call_args
        assert call_args[0][1] == 2  # 11 // 4 = 2

    def test_check_quota_estimates_list_tokens(self, settings: Settings) -> None:
        """_check_quota estimates tokens from list of Parts."""
        quota_checker = MagicMock()
        client = GeminiClient(settings, quota_checker=quota_checker)

        part_a = MagicMock()
        part_a.__str__ = lambda self: "a" * 100
        part_b = MagicMock()
        part_b.__str__ = lambda self: "b" * 100
        parts = [part_a, part_b]
        with patch("vaig.core.identity.resolve_identity", return_value=("u", "u@co.com", "1.0")):
            with patch("vaig.core.identity.build_composite_key", return_value="u:u@co.com"):
                client._check_quota(parts)

        call_args = quota_checker.check_and_consume.call_args
        assert call_args[0][1] == 50  # 200 // 4 = 50

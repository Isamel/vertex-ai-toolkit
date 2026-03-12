"""Tests for GeminiClient — google-genai SDK wrapper with mocked responses."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from vaig.core.client import ChatMessage, GeminiClient, GenerationResult
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelInfo,
    ModelsConfig,
    Settings,
    reset_settings,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset the settings singleton between tests."""
    reset_settings()


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


def _make_mock_stream_chunks(texts: list[str]) -> list[MagicMock]:
    """Create a list of mock streaming chunks."""
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.text = t
        chunks.append(chunk)
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
        mock_genai_client_cls.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=fake_creds,
        )
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

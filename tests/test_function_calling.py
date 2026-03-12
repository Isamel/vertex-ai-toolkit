"""Tests for function calling — ToolCallResult, build_function_response_parts, generate_with_tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.client import GeminiClient, ToolCallResult
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


# ── Helpers ──────────────────────────────────────────────────


def _make_function_call_part(name: str, args: dict | None = None) -> MagicMock:
    """Create a mock Part that contains a function call."""
    part = MagicMock()
    part.function_call.name = name
    part.function_call.args = args or {}
    part.text = ""  # no text for function call parts
    return part


def _make_text_part(text: str) -> MagicMock:
    """Create a mock Part that contains text only."""
    part = MagicMock()
    part.function_call = MagicMock()
    part.function_call.name = ""  # falsy name means not a function call
    part.text = text
    return part


def _make_tool_response(
    *,
    parts: list[MagicMock],
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    finish_reason: str = "STOP",
    has_usage: bool = True,
) -> MagicMock:
    """Create a mock Vertex AI response with configurable parts."""
    response = MagicMock()
    candidate = MagicMock()
    candidate.content.parts = parts
    candidate.finish_reason = finish_reason
    response.candidates = [candidate]

    if has_usage:
        response.usage_metadata.prompt_token_count = prompt_tokens
        response.usage_metadata.candidates_token_count = completion_tokens
        response.usage_metadata.total_token_count = total_tokens
    else:
        response.usage_metadata = None

    return response


def _make_empty_response(finish_reason: str = "STOP") -> MagicMock:
    """Create a mock response with no candidates or no parts."""
    response = MagicMock()
    response.candidates = []
    response.usage_metadata = None
    return response


# ===========================================================================
# ToolCallResult dataclass tests
# ===========================================================================


class TestToolCallResult:
    """Tests for the ToolCallResult dataclass."""

    def test_basic_construction_text_only(self) -> None:
        result = ToolCallResult(text="Hello", model="gemini-2.5-pro")
        assert result.text == "Hello"
        assert result.model == "gemini-2.5-pro"
        assert result.function_calls == []
        assert result.usage == {}
        assert result.finish_reason == ""

    def test_construction_with_function_calls(self) -> None:
        fcs = [
            {"name": "read_file", "args": {"path": "/tmp/test.py"}},
            {"name": "write_file", "args": {"path": "/tmp/out.py", "content": "x=1"}},
        ]
        result = ToolCallResult(
            text="",
            model="gemini-2.5-flash",
            function_calls=fcs,
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            finish_reason="STOP",
        )
        assert result.text == ""
        assert len(result.function_calls) == 2
        assert result.function_calls[0]["name"] == "read_file"
        assert result.function_calls[1]["args"]["content"] == "x=1"
        assert result.usage["total_tokens"] == 15
        assert result.finish_reason == "STOP"

    def test_default_function_calls_not_shared(self) -> None:
        """Each instance gets its own list (no mutable default sharing)."""
        r1 = ToolCallResult(text="a", model="m")
        r2 = ToolCallResult(text="b", model="m")
        r1.function_calls.append({"name": "foo", "args": {}})
        assert r2.function_calls == []

    def test_default_usage_not_shared(self) -> None:
        """Each instance gets its own dict."""
        r1 = ToolCallResult(text="a", model="m")
        r2 = ToolCallResult(text="b", model="m")
        r1.usage["prompt_tokens"] = 99
        assert r2.usage == {}

    def test_construction_with_all_fields(self) -> None:
        result = ToolCallResult(
            text="Done",
            model="gemini-2.5-pro",
            function_calls=[{"name": "list_files", "args": {"dir": "."}}],
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            finish_reason="MAX_TOKENS",
        )
        assert result.text == "Done"
        assert result.finish_reason == "MAX_TOKENS"
        assert result.usage["prompt_tokens"] == 100


# ===========================================================================
# build_function_response_parts() tests
# ===========================================================================


class TestBuildFunctionResponseParts:
    """Tests for GeminiClient.build_function_response_parts()."""

    @patch("vaig.core.client.Part.from_function_response")
    def test_single_result(self, mock_from_fn_response: MagicMock) -> None:
        mock_from_fn_response.return_value = MagicMock(name="part")

        results = [
            {
                "name": "read_file",
                "response": {"output": "file contents", "error": False},
            },
        ]

        parts = GeminiClient.build_function_response_parts(results)

        assert len(parts) == 1
        mock_from_fn_response.assert_called_once_with(
            name="read_file",
            response={"output": "file contents", "error": False},
        )

    @patch("vaig.core.client.Part.from_function_response")
    def test_multiple_results(self, mock_from_fn_response: MagicMock) -> None:
        mock_from_fn_response.side_effect = [MagicMock(), MagicMock(), MagicMock()]

        results = [
            {"name": "read_file", "response": {"output": "content", "error": False}},
            {"name": "write_file", "response": {"output": "OK", "error": False}},
            {"name": "run_command", "response": {"output": "exit 0", "error": False}},
        ]

        parts = GeminiClient.build_function_response_parts(results)

        assert len(parts) == 3
        assert mock_from_fn_response.call_count == 3

        # Verify each call
        calls = mock_from_fn_response.call_args_list
        assert calls[0].kwargs["name"] == "read_file"
        assert calls[1].kwargs["name"] == "write_file"
        assert calls[2].kwargs["name"] == "run_command"

    @patch("vaig.core.client.Part.from_function_response")
    def test_error_result(self, mock_from_fn_response: MagicMock) -> None:
        mock_from_fn_response.return_value = MagicMock()

        results = [
            {
                "name": "edit_file",
                "response": {"output": "File not found: /tmp/nope.py", "error": True},
            },
        ]

        parts = GeminiClient.build_function_response_parts(results)

        assert len(parts) == 1
        mock_from_fn_response.assert_called_once_with(
            name="edit_file",
            response={"output": "File not found: /tmp/nope.py", "error": True},
        )

    @patch("vaig.core.client.Part.from_function_response")
    def test_empty_results_list(self, mock_from_fn_response: MagicMock) -> None:
        parts = GeminiClient.build_function_response_parts([])

        assert parts == []
        mock_from_fn_response.assert_not_called()

    def test_is_static_method(self) -> None:
        """build_function_response_parts should be callable without an instance."""
        assert callable(GeminiClient.build_function_response_parts)


# ===========================================================================
# generate_with_tools() tests
# ===========================================================================


class TestGenerateWithTools:
    """Tests for GeminiClient.generate_with_tools()."""

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_text_response_no_function_calls(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When the model returns only text, result has text and no function_calls."""
        mock_get_creds.return_value = MagicMock()
        text_part = _make_text_part("Here is my analysis.")
        response = _make_tool_response(parts=[text_part])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Analyze this code",
            tool_declarations=[MagicMock()],
        )

        assert isinstance(result, ToolCallResult)
        assert result.text == "Here is my analysis."
        assert result.function_calls == []
        assert result.model == "gemini-2.5-pro"
        assert result.finish_reason == "STOP"
        assert result.usage["prompt_tokens"] == 10

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_function_call_response(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When the model returns function calls, they are parsed into dicts."""
        mock_get_creds.return_value = MagicMock()
        fc_part = _make_function_call_part("read_file", {"path": "/src/main.py"})
        response = _make_tool_response(parts=[fc_part])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Read the main file",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "read_file"
        assert result.function_calls[0]["args"]["path"] == "/src/main.py"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_multiple_function_calls_in_one_response(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Model can return multiple function calls in one response."""
        mock_get_creds.return_value = MagicMock()
        fc1 = _make_function_call_part("read_file", {"path": "a.py"})
        fc2 = _make_function_call_part("read_file", {"path": "b.py"})
        response = _make_tool_response(parts=[fc1, fc2])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Read both files",
            tool_declarations=[MagicMock()],
        )

        assert len(result.function_calls) == 2
        assert result.function_calls[0]["args"]["path"] == "a.py"
        assert result.function_calls[1]["args"]["path"] == "b.py"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_mixed_text_and_function_calls(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Response with both text and function calls."""
        mock_get_creds.return_value = MagicMock()
        text_part = _make_text_part("Let me read that file first.")
        fc_part = _make_function_call_part("read_file", {"path": "x.py"})
        response = _make_tool_response(parts=[text_part, fc_part])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Read x.py",
            tool_declarations=[MagicMock()],
        )

        assert result.text == "Let me read that file first."
        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "read_file"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_empty_response_no_candidates(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Empty response (no candidates) returns empty ToolCallResult."""
        mock_get_creds.return_value = MagicMock()
        response = _make_empty_response()
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert result.function_calls == []
        assert result.finish_reason == ""

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_empty_response_no_parts(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Response with candidate but no parts returns empty ToolCallResult."""
        mock_get_creds.return_value = MagicMock()
        response = MagicMock()
        candidate = MagicMock()
        candidate.content.parts = []
        candidate.finish_reason = "STOP"
        response.candidates = [candidate]
        response.usage_metadata = None
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert result.function_calls == []
        assert result.finish_reason == "STOP"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_usage_metadata_extraction(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Usage metadata is correctly extracted from response."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(
            parts=[_make_text_part("ok")],
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
        }

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_no_usage_metadata(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Missing usage metadata results in empty dict."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")], has_usage=False)
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.usage == {}

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_function_call_with_no_args(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Function call with None args results in empty dict."""
        mock_get_creds.return_value = MagicMock()
        fc_part = MagicMock()
        fc_part.function_call.name = "list_files"
        fc_part.function_call.args = None  # No args
        fc_part.text = ""
        response = _make_tool_response(parts=[fc_part])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools("List files", tool_declarations=[MagicMock()])

        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "list_files"
        assert result.function_calls[0]["args"] == {}

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Model override is respected in the result."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("fast response")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            model_id="gemini-2.5-flash",
        )

        assert result.model == "gemini-2.5-flash"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate_with_tools auto-initializes if not yet initialized."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        assert client._initialized is False
        client.generate_with_tools("Hello", tool_declarations=[MagicMock()])
        assert client._initialized is True
        mock_vertexai_init.assert_called_once()

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_tool_declarations_wrapped_in_tool(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Tool declarations are wrapped in a Tool() object for the API call."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        declarations = [MagicMock(name="decl_1"), MagicMock(name="decl_2")]
        client.generate_with_tools("Hello", tool_declarations=declarations)

        mock_tool_cls.assert_called_once_with(function_declarations=declarations)

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_with_system_instruction(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """System instruction is passed to _get_or_create_model."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            system_instruction="You are a coding expert.",
        )

        mock_model_cls.assert_called_once_with(
            "gemini-2.5-pro",
            system_instruction="You are a coding expert.",
        )

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_with_history_uses_chat(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When history is provided, uses start_chat + send_message."""
        mock_get_creds.return_value = MagicMock()
        from vaig.core.client import ChatMessage

        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        mock_model_cls.return_value = mock_model

        history = [ChatMessage(role="user", content="Previous question")]

        result = client.generate_with_tools(
            "Follow up",
            tool_declarations=[MagicMock()],
            history=history,
        )

        mock_model.start_chat.assert_called_once()
        mock_chat.send_message.assert_called_once()
        assert result.text == "ok"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_without_history_uses_generate_content(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Without history, uses model.generate_content directly."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("direct")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools(
            "No history",
            tool_declarations=[MagicMock()],
        )

        mock_model.generate_content.assert_called_once()
        mock_model.start_chat.assert_not_called()
        assert result.text == "direct"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_multiple_text_parts_concatenated(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Multiple text parts are joined together."""
        mock_get_creds.return_value = MagicMock()
        parts = [_make_text_part("Hello "), _make_text_part("World")]
        response = _make_tool_response(parts=parts)
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.text == "Hello World"

    @patch("vaig.core.client.Tool")
    @patch("vaig.core.client.GenerativeModel")
    @patch("vaig.core.client.GenerationConfig")
    @patch("vaig.core.client.vertexai.init")
    @patch("vaig.core.client.get_credentials")
    def test_gen_kwargs_forwarded(
        self,
        mock_get_creds: MagicMock,
        mock_vertexai_init: MagicMock,
        mock_gen_config_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Generation kwargs (temperature, etc.) are forwarded to config."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_model = MagicMock()
        mock_model.generate_content.return_value = response
        mock_model_cls.return_value = mock_model

        client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            temperature=0.2,
            max_output_tokens=4096,
        )

        mock_gen_config_cls.assert_called_once_with(
            temperature=0.2,
            max_output_tokens=4096,
            top_p=0.95,
            top_k=40,
        )

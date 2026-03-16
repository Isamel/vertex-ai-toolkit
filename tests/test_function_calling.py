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
    """Create a mock google-genai response with configurable parts."""
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

    @patch("vaig.core.client.types.Part.from_function_response")
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

    @patch("vaig.core.client.types.Part.from_function_response")
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

    @patch("vaig.core.client.types.Part.from_function_response")
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

    @patch("vaig.core.client.types.Part.from_function_response")
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
    """Tests for GeminiClient.generate_with_tools().

    Every test must patch ``types.Tool`` because the real Pydantic model
    rejects MagicMock tool_declarations.
    """

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_text_response_no_function_calls(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When the model returns only text, result has text and no function_calls."""
        mock_get_creds.return_value = MagicMock()
        text_part = _make_text_part("Here is my analysis.")
        response = _make_tool_response(parts=[text_part])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

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

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_function_call_response(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When the model returns function calls, they are parsed into dicts."""
        mock_get_creds.return_value = MagicMock()
        fc_part = _make_function_call_part("read_file", {"path": "/src/main.py"})
        response = _make_tool_response(parts=[fc_part])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Read the main file",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "read_file"
        assert result.function_calls[0]["args"]["path"] == "/src/main.py"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_multiple_function_calls_in_one_response(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Model can return multiple function calls in one response."""
        mock_get_creds.return_value = MagicMock()
        fc1 = _make_function_call_part("read_file", {"path": "a.py"})
        fc2 = _make_function_call_part("read_file", {"path": "b.py"})
        response = _make_tool_response(parts=[fc1, fc2])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Read both files",
            tool_declarations=[MagicMock()],
        )

        assert len(result.function_calls) == 2
        assert result.function_calls[0]["args"]["path"] == "a.py"
        assert result.function_calls[1]["args"]["path"] == "b.py"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_mixed_text_and_function_calls(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Response with both text and function calls."""
        mock_get_creds.return_value = MagicMock()
        text_part = _make_text_part("Let me read that file first.")
        fc_part = _make_function_call_part("read_file", {"path": "x.py"})
        response = _make_tool_response(parts=[text_part, fc_part])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Read x.py",
            tool_declarations=[MagicMock()],
        )

        assert result.text == "Let me read that file first."
        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "read_file"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_response_no_candidates(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Empty response (no candidates) returns empty ToolCallResult."""
        mock_get_creds.return_value = MagicMock()
        response = _make_empty_response()
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert result.function_calls == []
        assert result.finish_reason == ""

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_response_no_parts(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
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
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
        )

        assert result.text == ""
        assert result.function_calls == []
        assert result.finish_reason == "STOP"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_usage_metadata_extraction(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
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
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "thinking_tokens": 0,
        }

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_no_usage_metadata(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Missing usage metadata results in empty dict."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")], has_usage=False)
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.usage == {}

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_function_call_with_no_args(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Function call with None args results in empty dict."""
        mock_get_creds.return_value = MagicMock()
        fc_part = MagicMock()
        fc_part.function_call.name = "list_files"
        fc_part.function_call.args = None  # No args
        fc_part.text = ""
        response = _make_tool_response(parts=[fc_part])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools("List files", tool_declarations=[MagicMock()])

        assert len(result.function_calls) == 1
        assert result.function_calls[0]["name"] == "list_files"
        assert result.function_calls[0]["args"] == {}

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_model_override(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Model override is respected in the result."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("fast response")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            model_id="gemini-2.5-flash",
        )

        assert result.model == "gemini-2.5-flash"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_auto_initializes(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """generate_with_tools auto-initializes if not yet initialized."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        assert client._initialized is False
        client.generate_with_tools("Hello", tool_declarations=[MagicMock()])
        assert client._initialized is True
        mock_genai_client_cls.assert_called_once()

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_tool_declarations_wrapped_in_tool(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Tool declarations are wrapped in a types.Tool() object for the API call."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        declarations = [MagicMock(name="decl_1"), MagicMock(name="decl_2")]
        client.generate_with_tools("Hello", tool_declarations=declarations)

        mock_tool_cls.assert_called_once_with(function_declarations=declarations)

    @patch("vaig.core.client.types.GenerateContentConfig")
    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_with_system_instruction(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """System instruction is passed to GenerateContentConfig in the new SDK."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            system_instruction="You are a coding expert.",
        )

        # In the new SDK, system_instruction is part of GenerateContentConfig
        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["system_instruction"] == "You are a coding expert."

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_with_history_uses_chat(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """When history is provided, uses chats.create + send_message."""
        mock_get_creds.return_value = MagicMock()
        from vaig.core.client import ChatMessage

        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        history = [ChatMessage(role="user", content="Previous question")]

        result = client.generate_with_tools(
            "Follow up",
            tool_declarations=[MagicMock()],
            history=history,
        )

        mock_genai.chats.create.assert_called_once()
        mock_chat.send_message.assert_called_once()
        assert result.text == "ok"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_without_history_uses_generate_content(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Without history, uses client.models.generate_content directly."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("direct")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "No history",
            tool_declarations=[MagicMock()],
        )

        mock_genai.models.generate_content.assert_called_once()
        mock_genai.chats.create.assert_not_called()
        assert result.text == "direct"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_multiple_text_parts_concatenated(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Multiple text parts are joined together."""
        mock_get_creds.return_value = MagicMock()
        parts = [_make_text_part("Hello "), _make_text_part("World")]
        response = _make_tool_response(parts=parts)
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools("Test", tool_declarations=[MagicMock()])

        assert result.text == "Hello World"

    @patch("vaig.core.client.types.GenerateContentConfig")
    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_gen_kwargs_forwarded(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        mock_gen_config_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Generation kwargs (temperature, etc.) are forwarded to config."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        client.generate_with_tools(
            "Hello",
            tool_declarations=[MagicMock()],
            temperature=0.2,
            max_output_tokens=4096,
        )

        # Verify GenerateContentConfig was called with merged kwargs
        call_kwargs = mock_gen_config_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_output_tokens"] == 4096
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["top_k"] == 40


# ===========================================================================
# Empty-prompt-with-history tests (iteration 2+ bug fix)
# ===========================================================================


class TestEmptyPromptWithHistory:
    """Tests for the empty-prompt-pop-history fix in generate_with_tools.

    On iteration 2+ of a tool-calling loop, the CodingAgent sends an empty
    prompt because the full context (including function call responses) lives
    in the history.  The SDK rejects empty prompts with
    ``TypeError: value must not be empty``.

    The fix: when ``prompt`` is falsy and ``history`` is non-empty, pop the
    last history entry and use its ``.parts`` as the actual message sent to
    ``send_message()``.
    """

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_string_prompt_pops_last_history_entry(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Empty string prompt with history → pops last entry, sends its parts."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("tool response")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        # Create mock Content objects to simulate what _build_history returns.
        # We patch _build_history because Content() rejects MagicMock parts.
        fn_response_parts = [MagicMock(name="fn_response_part")]
        mock_content_1 = MagicMock(name="content_user")
        mock_content_2 = MagicMock(name="content_model")
        mock_content_3 = MagicMock(name="content_fn_response")
        mock_content_3.parts = fn_response_parts

        with patch.object(
            GeminiClient,
            "_build_history",
            return_value=[mock_content_1, mock_content_2, mock_content_3],
        ):
            from vaig.core.client import ChatMessage

            history = [
                ChatMessage(role="user", content="Fix the bug"),
                ChatMessage(role="model", content="Let me read the file"),
                ChatMessage(role="user", content="fn response placeholder"),
            ]

            result = client.generate_with_tools(
                "",  # <-- Empty prompt (iteration 2+)
                tool_declarations=[MagicMock()],
                history=history,
            )

        # The last history entry should have been popped, its parts used as prompt
        mock_chat.send_message.assert_called_once()
        actual_prompt = mock_chat.send_message.call_args[0][0]
        assert actual_prompt == fn_response_parts
        assert result.text == "tool response"

        # chats.create should have received history WITHOUT the last entry (popped)
        create_kwargs = mock_genai.chats.create.call_args[1]
        assert len(create_kwargs["history"]) == 2

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_list_prompt_pops_last_history_entry(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Empty list [] prompt with history → same pop behavior."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("done")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        fn_response_parts = [MagicMock(name="fn_response_part")]
        mock_content_1 = MagicMock(name="content_user")
        mock_content_2 = MagicMock(name="content_fn_response")
        mock_content_2.parts = fn_response_parts

        with patch.object(
            GeminiClient,
            "_build_history",
            return_value=[mock_content_1, mock_content_2],
        ):
            from vaig.core.client import ChatMessage

            history = [
                ChatMessage(role="user", content="Run tests"),
                ChatMessage(role="user", content="fn response placeholder"),
            ]

            result = client.generate_with_tools(
                [],  # <-- Empty list (CodingAgent sends this on iteration 2+)
                tool_declarations=[MagicMock()],
                history=history,
            )

        actual_prompt = mock_chat.send_message.call_args[0][0]
        assert actual_prompt == fn_response_parts
        assert result.text == "done"

        # History should have 1 entry (2nd was popped)
        create_kwargs = mock_genai.chats.create.call_args[1]
        assert len(create_kwargs["history"]) == 1

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_nonempty_prompt_with_history_does_not_pop(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Non-empty prompt with history → uses prompt as-is, no popping."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("reply")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        mock_content_1 = MagicMock(name="content_user")
        mock_content_2 = MagicMock(name="content_model")

        with patch.object(
            GeminiClient,
            "_build_history",
            return_value=[mock_content_1, mock_content_2],
        ):
            from vaig.core.client import ChatMessage

            history = [
                ChatMessage(role="user", content="First message"),
                ChatMessage(role="model", content="First reply"),
            ]

            result = client.generate_with_tools(
                "Follow up question",  # <-- Non-empty: first iteration
                tool_declarations=[MagicMock()],
                history=history,
            )

        # prompt should be sent as-is
        actual_prompt = mock_chat.send_message.call_args[0][0]
        assert actual_prompt == "Follow up question"

        # History should have ALL 2 entries (nothing popped)
        create_kwargs = mock_genai.chats.create.call_args[1]
        assert len(create_kwargs["history"]) == 2
        assert result.text == "reply"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_prompt_without_history_uses_generate_content(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Empty prompt without history → falls through to generate_content (no pop)."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("ok")])
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.models.generate_content.return_value = response

        result = client.generate_with_tools(
            "",  # Empty prompt, but no history → no pop logic
            tool_declarations=[MagicMock()],
        )

        # Should use generate_content (no history branch)
        mock_genai.models.generate_content.assert_called_once()
        mock_genai.chats.create.assert_not_called()
        assert result.text == "ok"

    @patch("vaig.core.client.types.Tool")
    @patch("vaig.core.client.genai.Client")
    @patch("vaig.core.client.get_credentials")
    def test_empty_prompt_single_history_entry_pops_to_empty(
        self,
        mock_get_creds: MagicMock,
        mock_genai_client_cls: MagicMock,
        _mock_tool_cls: MagicMock,
        client: GeminiClient,
    ) -> None:
        """Edge case: single history entry + empty prompt → pops it, chat gets empty history."""
        mock_get_creds.return_value = MagicMock()
        response = _make_tool_response(parts=[_make_text_part("edge")])
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = response
        mock_genai = mock_genai_client_cls.return_value
        mock_genai.chats.create.return_value = mock_chat

        fn_parts = [MagicMock(name="the_only_part")]
        mock_content = MagicMock(name="content_fn")
        mock_content.parts = fn_parts

        with patch.object(
            GeminiClient,
            "_build_history",
            return_value=[mock_content],
        ):
            from vaig.core.client import ChatMessage

            history = [
                ChatMessage(role="user", content="fn response placeholder"),
            ]

            result = client.generate_with_tools(
                "",
                tool_declarations=[MagicMock()],
                history=history,
            )

        # The single entry was popped → chat history is empty
        create_kwargs = mock_genai.chats.create.call_args[1]
        assert len(create_kwargs["history"]) == 0

        # The popped entry's parts become the prompt
        actual_prompt = mock_chat.send_message.call_args[0][0]
        assert actual_prompt == fn_parts
        assert result.text == "edge"

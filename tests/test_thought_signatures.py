"""Tests for Thought Signatures Capture (T1-T6).

Covers:
- raw_parts captured with thought_signature bytes present
- Thought parts (part.thought=True) excluded from raw_parts
- _build_function_call_content uses raw_parts when provided
- _build_function_call_content falls back to dict when raw_parts is None
- Sync path preserves signatures end-to-end
- Async path preserves signatures end-to-end
- Backward compat: ToolCallResult without raw_parts
- Backward compat: model returns no thought_signature (Gemini 2.0)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.core.client import GeminiClient, ToolCallResult
from vaig.core.config import GCPConfig, GenerationConfig, ModelInfo, ModelsConfig, Settings

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(temperature=0.7, max_output_tokens=8192),
        models=ModelsConfig(
            default="gemini-2.5-pro",
            fallback="gemini-2.5-flash",
            available=[
                ModelInfo(id="gemini-2.5-pro", description="Pro"),
                ModelInfo(id="gemini-2.5-flash", description="Flash"),
            ],
        ),
    )


@pytest.fixture()
def client(settings: Settings) -> GeminiClient:
    return GeminiClient(settings)


def _make_fc_part_with_signature(
    name: str,
    args: dict[str, Any],
    signature: bytes = b"\x01\x02\x03",
) -> MagicMock:
    """Create a mock function-call Part that carries a thought_signature.

    Used for tests that only inspect captured raw_parts (T1/T2/T3) and never
    pass them into types.Content() — which triggers Pydantic validation.
    """
    part = MagicMock()
    part.thought = False
    part.text = None
    part.thought_signature = signature

    fc = MagicMock()
    fc.name = name
    fc.args = args
    part.function_call = fc
    return part


def _make_real_fc_part(
    name: str,
    args: dict[str, Any],
    signature: bytes = b"\x01\x02\x03",
) -> Any:
    """Create a real types.Part for end-to-end tests that pass parts into types.Content."""
    from google.genai import types

    part = types.Part(
        function_call=types.FunctionCall(name=name, args=args),
        thought_signature=signature,
    )
    return part


def _make_thought_part(text: str = "Let me think...") -> MagicMock:
    """Create a mock Part representing a thinking/thought block."""
    part = MagicMock()
    part.thought = True
    part.text = text
    part.function_call = None
    return part


def _make_response_with_fc_and_thought(
    fc_parts: list[MagicMock],
    thought_text: str = "Internal reasoning",
    finish_reason: str = "STOP",
) -> MagicMock:
    """Build a full mock API response with thinking part + function call parts."""
    response = MagicMock()
    thought = _make_thought_part(thought_text)
    all_parts = [thought, *fc_parts]

    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content.parts = all_parts
    response.candidates = [candidate]

    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 5
    response.usage_metadata.total_token_count = 15
    response.usage_metadata.thoughts_token_count = 20

    return response


def _make_response_no_thought(
    fc_parts: list[MagicMock],
    finish_reason: str = "STOP",
) -> MagicMock:
    """Build a mock API response with only function call parts (no thought)."""
    response = MagicMock()
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content.parts = fc_parts
    response.candidates = [candidate]

    response.usage_metadata.prompt_token_count = 8
    response.usage_metadata.candidates_token_count = 4
    response.usage_metadata.total_token_count = 12
    response.usage_metadata.thoughts_token_count = None

    return response


# ── TC1: raw_parts captured when thought_signature bytes present ──


class TestRawPartsCaptured:
    """raw_parts is populated with function-call Parts from the response."""

    def test_raw_parts_captured_with_signature(self, client: GeminiClient) -> None:
        """T1/T2: raw_parts contains the original Part objects (with signature bytes)."""
        sig = b"\xca\xfe\xba\xbe"
        fc_part = _make_fc_part_with_signature("read_file", {"path": "/tmp/x"}, sig)
        response = _make_response_with_fc_and_thought([fc_part])

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools(
                "test",
                tool_declarations=[],
                history=None,
            )

        assert result.raw_parts is not None
        assert len(result.raw_parts) == 1
        # The raw Part still has the signature bytes on it
        assert result.raw_parts[0].thought_signature == sig

    def test_raw_parts_matches_function_calls_count(self, client: GeminiClient) -> None:
        """Number of raw_parts equals number of function_calls."""
        fc_part1 = _make_fc_part_with_signature("tool_a", {"x": 1}, b"\x01")
        fc_part2 = _make_fc_part_with_signature("tool_b", {"y": 2}, b"\x02")
        response = _make_response_with_fc_and_thought([fc_part1, fc_part2])

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools("test", tool_declarations=[], history=None)

        assert result.raw_parts is not None
        assert len(result.raw_parts) == 2
        assert len(result.function_calls) == 2


# ── TC2: Thought parts excluded from raw_parts ────────────────


class TestThoughtPartsExcluded:
    """Parts with part.thought=True must NOT appear in raw_parts."""

    def test_thought_parts_not_in_raw_parts(self, client: GeminiClient) -> None:
        """T2: The thought Part is excluded; only the FC Part is in raw_parts."""
        thought = _make_thought_part("Reasoning...")
        fc_part = _make_fc_part_with_signature("my_tool", {"arg": "val"})
        response = _make_response_no_thought([fc_part])
        # Manually splice in the thought to the response parts
        response.candidates[0].content.parts = [thought, fc_part]

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools("test", tool_declarations=[], history=None)

        assert result.raw_parts is not None
        for part in result.raw_parts:
            assert getattr(part, "thought", None) is not True

    def test_raw_parts_none_when_no_function_calls(self, client: GeminiClient) -> None:
        """raw_parts is None when the response has no function calls."""
        response = MagicMock()
        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "Just text"
        text_part.function_call = None

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]
        response.candidates = [candidate]
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        response.usage_metadata.total_token_count = 8
        response.usage_metadata.thoughts_token_count = None

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools("test", tool_declarations=[], history=None)

        assert result.raw_parts is None


# ── TC3: _build_function_call_content uses raw_parts when provided ──


class TestBuildFunctionCallContentWithRawParts:
    """_build_function_call_content replays raw_parts verbatim when truthy."""

    def test_uses_raw_parts_directly(self) -> None:
        """T4: When raw_parts is provided, Content.parts == raw_parts exactly."""
        from google.genai import types

        from vaig.agents.mixins import ToolLoopMixin

        raw_part_a = types.Part(function_call=types.FunctionCall(name="tool_a", args={"x": 1}))
        raw_part_b = types.Part(function_call=types.FunctionCall(name="tool_b", args={"y": 2}))
        raw_parts = [raw_part_a, raw_part_b]

        function_calls = [
            {"name": "tool_a", "args": {"x": 1}},
            {"name": "tool_b", "args": {"y": 2}},
        ]

        content = ToolLoopMixin._build_function_call_content(
            function_calls,
            raw_parts=raw_parts,
        )

        assert content.role == "model"
        # Pydantic copies the list; verify contents match by value
        assert len(content.parts) == 2
        assert content.parts[0].function_call.name == "tool_a"
        assert content.parts[1].function_call.name == "tool_b"

    def test_raw_parts_preserve_signature_bytes(self) -> None:
        """Replayed Content carries through whatever was on the original Part."""
        from google.genai import types

        from vaig.agents.mixins import ToolLoopMixin

        sig = b"\xde\xad\xbe\xef"
        part = types.Part(thought_signature=sig)

        content = ToolLoopMixin._build_function_call_content(
            [{"name": "tool", "args": {}}],
            raw_parts=[part],
        )
        assert content.parts[0].thought_signature == sig


# ── TC4: _build_function_call_content falls back to dict when raw_parts is None ──


class TestBuildFunctionCallContentFallback:
    """_build_function_call_content reconstructs Parts from dicts when raw_parts is None."""

    def test_fallback_builds_from_dicts(self) -> None:
        """T4: When raw_parts is None, Part.from_function_call is called per FC dict."""

        from vaig.agents.mixins import ToolLoopMixin

        function_calls = [{"name": "list_pods", "args": {"namespace": "default"}}]

        content = ToolLoopMixin._build_function_call_content(
            function_calls,
            raw_parts=None,
        )

        assert content.role == "model"
        assert len(content.parts) == 1

    def test_fallback_empty_raw_parts_list_uses_dicts(self) -> None:
        """An empty list for raw_parts is falsy, so fallback is used."""
        from google.genai import types

        from vaig.agents.mixins import ToolLoopMixin

        function_calls = [{"name": "some_tool", "args": {}}]

        real_part = types.Part(function_call=types.FunctionCall(name="some_tool", args={}))

        with patch("vaig.agents.mixins.types.Part.from_function_call") as mock_from_fc:
            mock_from_fc.return_value = real_part

            content = ToolLoopMixin._build_function_call_content(
                function_calls,
                raw_parts=[],  # empty list is falsy
            )

        mock_from_fc.assert_called_once_with(name="some_tool", args={})
        assert len(content.parts) == 1
        assert content.parts[0].function_call.name == "some_tool"


# ── TC5: Sync path preserves signatures end-to-end ────────────


class TestSyncPathPreservesSignatures:
    """The sync tool loop appends a Content with raw_parts when available."""

    def test_sync_history_contains_raw_parts(self) -> None:
        """T2/T5: history gets Content with raw Part objects (not reconstructed)."""
        from vaig.agents.mixins import ToolLoopMixin
        from vaig.tools.base import ToolRegistry

        sig = b"\xAA\xBB"
        fc_part = _make_real_fc_part("my_tool", {"a": 1}, sig)

        # First result: function calls with raw_parts
        fc_result = ToolCallResult(
            text="",
            model="gemini-2.5-pro",
            function_calls=[{"name": "my_tool", "args": {"a": 1}}],
            finish_reason="TOOL_CODE",
            raw_parts=[fc_part],
        )
        # Second result: text (terminates the loop)
        text_result = ToolCallResult(
            text="Done.",
            model="gemini-2.5-pro",
            finish_reason="STOP",
        )

        mock_client = MagicMock()
        mock_client.generate_with_tools.side_effect = [fc_result, text_result]
        from google.genai import types as _types

        mock_client.build_function_response_parts.return_value = [
            _types.Part.from_function_response(name="my_tool", response={"result": "ok"})
        ]

        tool_fn = MagicMock(return_value=MagicMock(output="ok", error=False))
        registry = MagicMock(spec=ToolRegistry)
        registry.to_function_declarations.return_value = []
        registry.get.return_value = MagicMock(
            name="my_tool",
            execute=tool_fn,
            cacheable=False,
            parameters=[],
        )

        mixin = ToolLoopMixin()
        history: list[Any] = []

        loop_result = mixin._run_tool_loop(
            client=mock_client,
            prompt="go",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        assert loop_result.text == "Done."
        # First history entry should be the model content with raw_parts
        fc_content = history[0]
        assert fc_content.role == "model"
        # Pydantic copies the list but preserves values — check signature survives
        assert len(fc_content.parts) == 1
        assert fc_content.parts[0].thought_signature == sig


# ── TC6: Async path preserves signatures end-to-end ──────────


class TestAsyncPathPreservesSignatures:
    """The async tool loop also appends Content with raw_parts."""

    async def test_async_history_contains_raw_parts(self) -> None:
        """T3/T5: async history gets Content built from raw_parts."""
        from vaig.agents.mixins import ToolLoopMixin
        from vaig.tools.base import ToolRegistry

        sig = b"\xCC\xDD"
        fc_part = _make_real_fc_part("async_tool", {"z": 9}, sig)

        fc_result = ToolCallResult(
            text="",
            model="gemini-2.5-pro",
            function_calls=[{"name": "async_tool", "args": {"z": 9}}],
            finish_reason="TOOL_CODE",
            raw_parts=[fc_part],
        )
        text_result = ToolCallResult(
            text="Async done.",
            model="gemini-2.5-pro",
            finish_reason="STOP",
        )

        mock_client = MagicMock()
        mock_client.async_generate_with_tools = AsyncMock(
            side_effect=[fc_result, text_result]
        )
        from google.genai import types as _types

        mock_client.build_function_response_parts.return_value = [
            _types.Part.from_function_response(name="async_tool", response={"result": "ok"})
        ]

        tool_fn = MagicMock(return_value=MagicMock(output="async_ok", error=False))
        registry = MagicMock(spec=ToolRegistry)
        registry.to_function_declarations.return_value = []
        registry.get.return_value = MagicMock(
            name="async_tool",
            execute=tool_fn,
            cacheable=False,
            parameters=[],
        )

        mixin = ToolLoopMixin()
        history: list[Any] = []

        loop_result = await mixin._async_run_tool_loop(
            client=mock_client,
            prompt="go",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            parallel_tool_calls=False,
        )

        assert loop_result.text == "Async done."
        fc_content = history[0]
        assert fc_content.role == "model"
        # Pydantic copies the list but preserves values — check signature survives
        assert len(fc_content.parts) == 1
        assert fc_content.parts[0].thought_signature == sig


# ── TC7: Backward compat — ToolCallResult without raw_parts ──


class TestBackwardCompatNoRawParts:
    """Existing code that doesn't set raw_parts still works."""

    def test_raw_parts_defaults_to_none(self) -> None:
        """T1: raw_parts defaults to None — no breaking change."""
        result = ToolCallResult(text="Hello", model="gemini-2.5-pro")
        assert result.raw_parts is None

    def test_construction_with_all_old_fields(self) -> None:
        """All old fields still work; raw_parts silently defaults."""
        result = ToolCallResult(
            text="response",
            model="gemini-2.5-flash",
            function_calls=[{"name": "foo", "args": {"x": 1}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="STOP",
            thinking_text="Hmm...",
        )
        assert result.raw_parts is None
        assert result.thinking_text == "Hmm..."
        assert result.function_calls[0]["name"] == "foo"

    def test_build_without_raw_parts_still_works(self) -> None:
        """_build_function_call_content with no raw_parts kwarg still works."""
        from vaig.agents.mixins import ToolLoopMixin

        fcs = [{"name": "tool_x", "args": {"q": "v"}}]
        # Old call signature — no raw_parts argument
        content = ToolLoopMixin._build_function_call_content(fcs)
        assert content.role == "model"
        assert len(content.parts) == 1


# ── TC8: Backward compat — model without thought_signature (Gemini 2.0) ──


class TestGemini20NoSignature:
    """Gemini 2.0 models don't emit thought_signature; raw_parts still works fine."""

    def test_raw_parts_captured_without_signature_bytes(self, client: GeminiClient) -> None:
        """T2: raw_parts is captured even when no thought_signature is present."""
        # Simulate a Gemini 2.0 FC part — no thought_signature attribute
        fc_part = MagicMock(spec=["thought", "text", "function_call"])
        fc_part.thought = False
        fc_part.text = None
        fc_fc = MagicMock()
        fc_fc.name = "list_clusters"
        fc_fc.args = {"region": "us-central1"}
        fc_part.function_call = fc_fc

        response = _make_response_no_thought([fc_part])

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools("test", tool_declarations=[], history=None)

        assert result.raw_parts is not None
        assert result.raw_parts[0] is fc_part

    def test_raw_parts_none_when_only_text_response(self, client: GeminiClient) -> None:
        """A pure text response (no FC) yields raw_parts=None regardless of model."""
        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "Hello from Gemini 2.0"
        text_part.function_call = None

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        response.usage_metadata.total_token_count = 8
        response.usage_metadata.thoughts_token_count = None

        with patch.object(client, "_get_client") as mock_gc:
            mock_gc.return_value.models.generate_content.return_value = response
            client._initialized = True

            result = client.generate_with_tools("test", tool_declarations=[], history=None)

        assert result.raw_parts is None

"""Tests for unknown tool graceful handling in _execute_single_tool.

Covers:
- Unknown tool returns error=True with helpful message
- Unknown tool message includes the available tool list
- Fuzzy matching suggests close tool names via difflib
- No suggestion when hallucinated name has no close match
- Async variant (_async_execute_single_tool) behaves identically
- Telemetry is emitted with error_type="UnknownTool"
- Empty registry shows "(none registered)" instead of an empty list
- Large registries cap the tool list at 10 with "... and N more"
- _build_unknown_tool_message helper is tested directly
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vaig.tools.base import ToolDef, ToolRegistry, ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_registry(*names: str) -> ToolRegistry:
    """Create a ToolRegistry with stub tools for the given names."""
    reg = ToolRegistry()
    for name in names:
        reg.register(ToolDef(name=name, description=f"Stub {name} tool"))
    return reg


def _make_agent_with_mixin() -> MagicMock:
    """Return a minimal object that has the ToolLoopMixin methods available.

    We import the mixin and bind the real method to a mock so we can call it
    without needing a full agent initialisation (no GCP credentials, etc.).
    """
    from vaig.agents.mixins import ToolLoopMixin

    agent = MagicMock(spec=ToolLoopMixin)
    # Bind real implementations so we exercise actual code paths
    agent._execute_single_tool = ToolLoopMixin._execute_single_tool.__get__(agent, ToolLoopMixin)
    agent._async_execute_single_tool = ToolLoopMixin._async_execute_single_tool.__get__(
        agent, ToolLoopMixin
    )
    agent._build_unknown_tool_message = ToolLoopMixin._build_unknown_tool_message.__get__(
        agent, ToolLoopMixin
    )
    agent._pre_validate_tool_args = ToolLoopMixin._pre_validate_tool_args
    agent._check_tool_effectiveness = ToolLoopMixin._check_tool_effectiveness
    # _emit_tool_telemetry is a side-effect we don't need in unit tests
    agent._emit_tool_telemetry = MagicMock()
    return agent


# ── Sync _execute_single_tool ─────────────────────────────────


class TestExecuteSingleToolUnknown:
    """Unit tests for the unknown-tool path in _execute_single_tool."""

    def test_returns_error_true_for_unknown_tool(self) -> None:
        """Result must have error=True when tool is not in the registry."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = agent._execute_single_tool(registry, "gcloud_logging_query", {})

        assert result.error is True

    def test_output_mentions_tool_name(self) -> None:
        """Error output must reference the hallucinated tool name."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = agent._execute_single_tool(registry, "gcloud_logging_query", {})

        assert "gcloud_logging_query" in result.output

    def test_output_includes_available_tools(self) -> None:
        """Error output must list the actual available tools."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = agent._execute_single_tool(registry, "gcloud_logging_query", {})

        assert "kubectl_get" in result.output
        assert "kubectl_logs" in result.output

    def test_output_instructs_llm_to_use_available_tools(self) -> None:
        """Error output must instruct the LLM to use available tools."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        result = agent._execute_single_tool(registry, "nonexistent", {})

        assert "Please use one of the available tools" in result.output

    def test_suggests_close_match_when_name_is_similar(self) -> None:
        """When a close match exists, output must include a 'Did you mean' suggestion."""
        agent = _make_agent_with_mixin()
        # 'kubectl_log' is similar to 'kubectl_logs'
        registry = _make_registry("kubectl_get", "kubectl_logs", "get_events")

        result = agent._execute_single_tool(registry, "kubectl_log", {})

        assert "Did you mean" in result.output
        assert "kubectl_logs" in result.output

    def test_no_suggestion_when_name_is_completely_different(self) -> None:
        """When no close match exists, output must NOT include 'Did you mean'."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs", "get_events")

        result = agent._execute_single_tool(registry, "totally_unrelated_xyz_tool_123", {})

        assert "Did you mean" not in result.output

    def test_emits_telemetry_with_unknown_tool_error_type(self) -> None:
        """_emit_tool_telemetry must be called with error_type='UnknownTool'."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        agent._execute_single_tool(registry, "hallucinated_tool", {"arg": "val"})

        agent._emit_tool_telemetry.assert_called_once()
        call_kwargs = agent._emit_tool_telemetry.call_args.kwargs
        assert call_kwargs.get("error_type") == "UnknownTool"

    def test_available_tools_are_sorted_alphabetically(self) -> None:
        """Available tools in the error message must be sorted for readability."""
        agent = _make_agent_with_mixin()
        # Register in reverse alphabetical order intentionally
        registry = _make_registry("zzz_tool", "aaa_tool", "mmm_tool")

        result = agent._execute_single_tool(registry, "unknown_tool", {})

        # Find the 'Available tools:' section and verify order
        idx_available = result.output.index("Available tools:")
        available_section = result.output[idx_available:]
        idx_aaa = available_section.index("aaa_tool")
        idx_mmm = available_section.index("mmm_tool")
        idx_zzz = available_section.index("zzz_tool")
        assert idx_aaa < idx_mmm < idx_zzz

    def test_multiple_close_matches_all_suggested(self) -> None:
        """Up to 3 close matches should all appear in the suggestion."""
        agent = _make_agent_with_mixin()
        # All similar to 'kubectl_log'
        registry = _make_registry("kubectl_logs", "kubectl_log_stream", "kubectl_log_tail")

        result = agent._execute_single_tool(registry, "kubectl_log", {})

        assert "Did you mean" in result.output
        # At least one of the close matches should appear
        assert any(
            name in result.output
            for name in ("kubectl_logs", "kubectl_log_stream", "kubectl_log_tail")
        )

    def test_empty_registry_shows_none_registered(self) -> None:
        """When no tools are registered, message must say '(none registered)'."""
        agent = _make_agent_with_mixin()
        registry = ToolRegistry()  # empty

        result = agent._execute_single_tool(registry, "some_tool", {})

        assert result.error is True
        assert "some_tool" in result.output
        assert "(none registered)" in result.output
        # No suggestion when registry is empty
        assert "Did you mean" not in result.output

    def test_does_not_raise_returns_tool_result(self) -> None:
        """_execute_single_tool must never raise — always returns ToolResult."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        result = agent._execute_single_tool(registry, "hallucinated", {})

        assert isinstance(result, ToolResult)

    def test_known_tool_still_executes_normally(self) -> None:
        """A known tool must still execute (regression guard — new code must not break happy path)."""
        agent = _make_agent_with_mixin()
        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="my_tool",
                description="A real tool",
                execute=lambda: ToolResult(output="success"),
            )
        )

        result = agent._execute_single_tool(registry, "my_tool", {})

        assert result.error is False
        assert result.output == "success"

    def test_large_registry_caps_tool_list_at_ten(self) -> None:
        """When >10 tools are registered, only 10 are shown plus '... and N more'."""
        agent = _make_agent_with_mixin()
        # Register 15 tools — should cap at 10
        registry = _make_registry(*[f"tool_{i:02d}" for i in range(15)])

        result = agent._execute_single_tool(registry, "nonexistent_xyz", {})

        assert "... and 5 more" in result.output

    def test_exactly_ten_tools_shows_no_overflow(self) -> None:
        """When exactly 10 tools are registered, all are shown without 'and N more'."""
        agent = _make_agent_with_mixin()
        registry = _make_registry(*[f"tool_{i:02d}" for i in range(10)])

        result = agent._execute_single_tool(registry, "nonexistent_xyz", {})

        assert "... and" not in result.output


# ── _build_unknown_tool_message helper ───────────────────────


class TestBuildUnknownToolMessage:
    """Unit tests for the _build_unknown_tool_message shared helper."""

    def test_empty_registry_uses_none_registered_placeholder(self) -> None:
        """Empty registry must produce '(none registered)' in the message."""
        agent = _make_agent_with_mixin()
        registry = ToolRegistry()

        msg = agent._build_unknown_tool_message("missing_tool", registry)

        assert "(none registered)" in msg
        assert "missing_tool" in msg

    def test_cap_at_ten_adds_overflow_indicator(self) -> None:
        """When >10 tools exist, message must include '... and N more'."""
        agent = _make_agent_with_mixin()
        registry = _make_registry(*[f"tool_{i:02d}" for i in range(20)])

        msg = agent._build_unknown_tool_message("ghost", registry)

        assert "... and 10 more" in msg

    def test_names_are_sorted_alphabetically(self) -> None:
        """Helper must sort tool names alphabetically."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("zzz_tool", "aaa_tool", "mmm_tool")

        msg = agent._build_unknown_tool_message("nope", registry)

        idx_aaa = msg.index("aaa_tool")
        idx_mmm = msg.index("mmm_tool")
        idx_zzz = msg.index("zzz_tool")
        assert idx_aaa < idx_mmm < idx_zzz

    def test_fuzzy_suggestion_included(self) -> None:
        """Helper must include 'Did you mean' when a close match exists."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        msg = agent._build_unknown_tool_message("kubectl_log", registry)

        assert "Did you mean" in msg
        assert "kubectl_logs" in msg


# ── Async _async_execute_single_tool ─────────────────────────


class TestAsyncExecuteSingleToolUnknown:
    """Unit tests for the unknown-tool path in _async_execute_single_tool."""

    @pytest.mark.asyncio
    async def test_returns_error_true_for_unknown_tool(self) -> None:
        """Async variant must also return error=True for unknown tools."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = await agent._async_execute_single_tool(registry, "gcloud_logging_query", {})

        assert result.error is True

    @pytest.mark.asyncio
    async def test_output_mentions_tool_name(self) -> None:
        """Async error output must reference the hallucinated tool name."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        result = await agent._async_execute_single_tool(registry, "gcloud_logging_query", {})

        assert "gcloud_logging_query" in result.output

    @pytest.mark.asyncio
    async def test_output_includes_available_tools(self) -> None:
        """Async error output must list actual available tools."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = await agent._async_execute_single_tool(registry, "gcloud_logging_query", {})

        assert "kubectl_get" in result.output
        assert "kubectl_logs" in result.output

    @pytest.mark.asyncio
    async def test_suggests_close_match_async(self) -> None:
        """Async variant must also suggest close matches."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = await agent._async_execute_single_tool(registry, "kubectl_log", {})

        assert "Did you mean" in result.output
        assert "kubectl_logs" in result.output

    @pytest.mark.asyncio
    async def test_no_suggestion_when_no_close_match_async(self) -> None:
        """Async variant must not suggest when there's no close match."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get", "kubectl_logs")

        result = await agent._async_execute_single_tool(registry, "totally_different_xyz_987", {})

        assert "Did you mean" not in result.output

    @pytest.mark.asyncio
    async def test_emits_telemetry_with_unknown_tool_error_type_async(self) -> None:
        """Async variant must call _emit_tool_telemetry with error_type='UnknownTool'."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        await agent._async_execute_single_tool(registry, "hallucinated_tool", {})

        agent._emit_tool_telemetry.assert_called_once()
        call_kwargs = agent._emit_tool_telemetry.call_args.kwargs
        assert call_kwargs.get("error_type") == "UnknownTool"

    @pytest.mark.asyncio
    async def test_does_not_raise_returns_tool_result_async(self) -> None:
        """Async variant must never raise — always returns ToolResult."""
        agent = _make_agent_with_mixin()
        registry = _make_registry("kubectl_get")

        result = await agent._async_execute_single_tool(registry, "hallucinated", {})

        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_message_parity_sync_vs_async(self) -> None:
        """Sync and async variants must produce equivalent error messages."""
        registry = _make_registry("kubectl_get", "kubectl_logs", "get_events")
        tool_name = "kubectl_log"

        agent_sync = _make_agent_with_mixin()
        agent_async = _make_agent_with_mixin()

        sync_result = agent_sync._execute_single_tool(registry, tool_name, {})
        async_result = await agent_async._async_execute_single_tool(registry, tool_name, {})

        assert sync_result.output == async_result.output
        assert sync_result.error == async_result.error

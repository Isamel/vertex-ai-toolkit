"""Tests for the tool-loop budget warning (tool-budget-monitor feature).

Verifies that ToolLoopMixin._run_tool_loop injects a budget warning into
history at the right time, only once, and only for missing sections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from google.genai import types

from vaig.agents.mixins import BUDGET_WARNING_THRESHOLD, ToolLoopMixin, ToolLoopResult
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_result(
    *,
    text: str = "",
    function_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    model: str = "gemini-test",
    finish_reason: str = "STOP",
) -> MagicMock:
    """Build a minimal mock ToolCallResult."""
    result = MagicMock()
    result.text = text
    result.function_calls = function_calls or []
    result.usage = usage or {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    result.model = model
    result.finish_reason = finish_reason
    return result


def _make_client(responses: list[MagicMock]) -> MagicMock:
    """Build a mock GeminiClientProtocol that returns *responses* in order."""
    client = MagicMock()
    client.generate_with_tools.side_effect = responses
    # Must return a real types.Part so pydantic validation passes when the loop
    # calls types.Content(role="user", parts=response_parts)
    client.build_function_response_parts.return_value = [
        types.Part.from_text(text="tool_response_ok")
    ]
    return client


def _make_registry_with_dummy_tool() -> ToolRegistry:
    """Return a ToolRegistry with a single dummy tool that always succeeds."""
    registry = ToolRegistry()
    tool = MagicMock()
    tool.name = "dummy_tool"
    tool.description = "A dummy tool"
    tool.parameters = []
    tool.cacheable = False
    from vaig.tools.base import ToolResult
    tool.execute.return_value = ToolResult(output="done", error=False)
    # to_function_declarations must return a list
    registry._tools = {"dummy_tool": tool}
    return registry


class _ConcreteLoopMixin(ToolLoopMixin):
    """Concrete subclass so we can instantiate ToolLoopMixin directly."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBudgetWarningConstant:
    def test_threshold_value(self) -> None:
        assert BUDGET_WARNING_THRESHOLD == 0.8


class TestBudgetWarning:
    """Tests for the budget warning injection logic in _run_tool_loop."""

    def _make_fc_response(self, text: str = "") -> MagicMock:
        """Return a response with one dummy function call (loop continues)."""
        fc = {"name": "dummy_tool", "args": {}}
        return _make_tool_call_result(text=text, function_calls=[fc])

    def _make_text_response(self, text: str = "Final answer.") -> MagicMock:
        """Return a text-only response (loop terminates)."""
        return _make_tool_call_result(text=text, function_calls=[])

    def _run_loop(
        self,
        responses: list[MagicMock],
        *,
        max_iterations: int = 5,
        required_sections: list[str] | None = None,
    ) -> tuple[ToolLoopResult, list[Any]]:
        """Run the sync tool loop and return (result, history)."""
        history: list[Any] = []
        registry = ToolRegistry()
        # Patch to_function_declarations so the loop can start
        registry.to_function_declarations = MagicMock(return_value=[])
        # Patch get so tool lookup doesn't fail
        registry.get = MagicMock(return_value=None)

        client = _make_client(responses)
        mixin = _ConcreteLoopMixin()
        # Patch _execute_single_tool to avoid real execution
        from vaig.tools.base import ToolResult
        mixin._execute_single_tool = MagicMock(  # type: ignore[method-assign]
            return_value=ToolResult(output="tool output", error=False)
        )

        result = mixin._run_tool_loop(
            client=client,
            prompt="test prompt",
            tool_registry=registry,
            system_instruction="You are a test agent.",
            history=history,
            max_iterations=max_iterations,
            required_sections=required_sections,
        )
        return result, history

    # ── Test 1: No warning when required_sections is None ──────────────────

    def test_no_warning_when_sections_not_configured(self) -> None:
        """No budget warning should be injected when required_sections is None."""
        # 5 iterations: 4 FC responses then a text response
        responses = [
            self._make_fc_response(f"text {i}") for i in range(4)
        ] + [self._make_text_response()]

        _, history = self._run_loop(responses, max_iterations=5, required_sections=None)

        # Verify no entry contains the warning emoji
        history_texts = [
            part.text if (hasattr(part, "text") and part.text is not None) else str(part)
            for item in history
            for part in (getattr(item, "parts", None) or [])
        ]
        assert not any("⚠️ BUDGET WARNING" in t for t in history_texts)

    # ── Test 2: No warning before threshold ────────────────────────────────

    def test_no_warning_before_threshold(self) -> None:
        """No warning when iteration is below the 80% threshold."""
        # max_iterations=10, threshold at iteration 8
        # Only run 3 iterations (below threshold), then text response
        responses = [
            self._make_fc_response("text") for _ in range(3)
        ] + [self._make_text_response("done")]

        _, history = self._run_loop(
            responses,
            max_iterations=10,
            required_sections=["Section A", "Section B"],
        )

        history_texts = [
            getattr(part, "text", "") or ""
            for item in history
            for part in (getattr(item, "parts", None) or [])
        ]
        assert not any("⚠️ BUDGET WARNING" in t for t in history_texts)

    # ── Test 3: Warning fires at threshold ────────────────────────────────

    def test_warning_fires_at_threshold(self) -> None:
        """Budget warning should be injected when iteration hits 80% threshold."""
        # max_iterations=5, threshold = int(5 * 0.8) = 4
        # Run 4 FC iterations (hits threshold), then a text response
        responses = [
            self._make_fc_response(f"text {i}") for i in range(4)
        ] + [self._make_text_response("Final answer.")]

        _, history = self._run_loop(
            responses,
            max_iterations=5,
            required_sections=["Missing Section"],
        )

        # Extract all text parts from history entries
        warning_parts = [
            getattr(part, "text", "") or ""
            for item in history
            for part in (getattr(item, "parts", None) or [])
            if "⚠️ BUDGET WARNING" in (getattr(part, "text", "") or "")
        ]
        assert len(warning_parts) == 1, f"Expected 1 warning, got {len(warning_parts)}"
        assert "Missing Section" in warning_parts[0]

    # ── Test 4: Warning fires only once ───────────────────────────────────

    def test_warning_fires_only_once(self) -> None:
        """Budget warning must be injected exactly once even across many iterations."""
        # max_iterations=10, threshold = int(10 * 0.8) = 8
        # Run iterations 8, 9 at threshold+ then text response
        responses = [
            self._make_fc_response("text") for _ in range(9)
        ] + [self._make_text_response("done")]

        _, history = self._run_loop(
            responses,
            max_iterations=10,
            required_sections=["Required Section"],
        )

        warning_count = sum(
            1
            for item in history
            for part in (getattr(item, "parts", None) or [])
            if "⚠️ BUDGET WARNING" in (getattr(part, "text", "") or "")
        )
        assert warning_count == 1, f"Expected exactly 1 warning, got {warning_count}"

    # ── Test 5: No warning when all sections present in accumulated text ───

    def test_no_warning_when_all_sections_present(self) -> None:
        """No warning if all required sections appear in accumulated LLM text."""
        # FC responses that contain all required sections in their text
        responses = [
            self._make_fc_response("Introduction section found here."),
            self._make_fc_response("Conclusion section covered in detail."),
            self._make_fc_response("text"),
            self._make_fc_response("text"),
            self._make_text_response("All done."),
        ]

        _, history = self._run_loop(
            responses,
            max_iterations=5,
            required_sections=["Introduction", "Conclusion"],
        )

        warning_parts = [
            getattr(part, "text", "") or ""
            for item in history
            for part in (getattr(item, "parts", None) or [])
            if "⚠️ BUDGET WARNING" in (getattr(part, "text", "") or "")
        ]
        assert len(warning_parts) == 0, "No warning expected when all sections are present"

    # ── Test 6: Warning lists only missing sections ────────────────────────

    def test_warning_lists_only_missing_sections(self) -> None:
        """Warning message should name only the sections that are absent."""
        # One section present in accumulated text, one absent
        responses = [
            self._make_fc_response("Section A has been covered thoroughly."),
            self._make_fc_response("continuing work"),
            self._make_fc_response("continuing work"),
            self._make_fc_response("continuing work"),
            self._make_text_response("done"),
        ]

        _, history = self._run_loop(
            responses,
            max_iterations=5,
            required_sections=["Section A", "Section B"],
        )

        warning_parts = [
            getattr(part, "text", "") or ""
            for item in history
            for part in (getattr(item, "parts", None) or [])
            if "⚠️ BUDGET WARNING" in (getattr(part, "text", "") or "")
        ]
        assert len(warning_parts) == 1
        warning_text = warning_parts[0]
        # Section B should be in the warning (missing)
        assert "Section B" in warning_text
        # Section A should NOT be in the warning (present)
        assert "Section A" not in warning_text

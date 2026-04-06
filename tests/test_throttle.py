"""Tests for RPM-aware inter-call throttling in ToolLoopMixin.

The ``min_inter_call_delay`` setting in ``AgentsConfig`` inserts a sleep
between tool-loop iterations to avoid hitting Vertex AI RPM quotas.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.core.config import AgentsConfig


class TestMinInterCallDelay:
    """Tests for the min_inter_call_delay config field."""

    def test_default_is_zero(self) -> None:
        """Default min_inter_call_delay is 0.0 (throttling disabled)."""
        cfg = AgentsConfig()
        assert cfg.min_inter_call_delay == 0.0

    def test_custom_value(self) -> None:
        """min_inter_call_delay accepts a custom float value."""
        cfg = AgentsConfig(min_inter_call_delay=2.5)
        assert cfg.min_inter_call_delay == 2.5


class TestThrottleCodePath:
    """Verify the throttle code path is wired correctly in ToolLoopMixin."""

    @patch("vaig.agents.mixins.time.sleep")
    @patch("vaig.agents.mixins.get_settings")
    def test_throttle_hoisted_from_settings(
        self,
        mock_get_settings: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """ToolLoopMixin reads min_inter_call_delay before entering the loop."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_settings = MagicMock()
        mock_settings.agents.min_inter_call_delay = 1.5
        mock_settings.context_window.warn_threshold_pct = 80.0
        mock_settings.context_window.error_threshold_pct = 95.0
        mock_get_settings.return_value = mock_settings

        mixin = ToolLoopMixin()

        # Client returns text immediately — 1 iteration only → no sleep
        mock_result = MagicMock()
        mock_result.text = "Done"
        mock_result.function_calls = []
        mock_result.tool_calls = []
        mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        mock_result.finish_reason = "STOP"

        mock_client = MagicMock()
        mock_client.generate_with_tools.return_value = mock_result

        mock_registry = MagicMock()
        mock_registry.to_function_declarations.return_value = []

        mixin._run_tool_loop(
            client=mock_client,
            prompt="test",
            tool_registry=mock_registry,
            system_instruction="",
            history=[],
            max_iterations=5,
        )

        # Only 1 iteration — throttle skips iteration 1
        mock_sleep.assert_not_called()

    @patch("vaig.agents.mixins.time.sleep")
    @patch("vaig.agents.mixins.get_settings")
    def test_no_throttle_when_delay_is_zero(
        self,
        mock_get_settings: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """time.sleep is NOT called when min_inter_call_delay is 0."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_settings = MagicMock()
        mock_settings.agents.min_inter_call_delay = 0.0
        mock_settings.context_window.warn_threshold_pct = 80.0
        mock_settings.context_window.error_threshold_pct = 95.0
        mock_get_settings.return_value = mock_settings

        mixin = ToolLoopMixin()

        mock_result = MagicMock()
        mock_result.text = "Done"
        mock_result.function_calls = []
        mock_result.tool_calls = []
        mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        mock_result.finish_reason = "STOP"

        mock_client = MagicMock()
        mock_client.generate_with_tools.return_value = mock_result

        mock_registry = MagicMock()
        mock_registry.to_function_declarations.return_value = []

        mixin._run_tool_loop(
            client=mock_client,
            prompt="test",
            tool_registry=mock_registry,
            system_instruction="",
            history=[],
            max_iterations=5,
        )

        mock_sleep.assert_not_called()

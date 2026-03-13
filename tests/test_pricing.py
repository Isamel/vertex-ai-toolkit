"""Tests for the pricing module — cost calculation and formatting."""

from __future__ import annotations

import pytest

from vaig.core.pricing import MODEL_PRICING, ModelPricing, calculate_cost, format_cost


class TestCalculateCost:
    """Tests for calculate_cost()."""

    def test_known_model_returns_correct_cost(self) -> None:
        """Gemini 2.5 Pro: 1000 prompt + 500 completion tokens."""
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        assert cost is not None
        # input: (1000 / 1_000_000) * 1.25 = 0.00125
        # output: (500 / 1_000_000) * 10.00 = 0.005
        # total: 0.00625
        assert cost == pytest.approx(0.00625)

    def test_flash_model_returns_correct_cost(self) -> None:
        """Gemini 2.5 Flash: cheaper pricing."""
        cost = calculate_cost("gemini-2.5-flash", prompt_tokens=10_000, completion_tokens=2_000)
        assert cost is not None
        # input: (10000 / 1_000_000) * 0.15 = 0.0015
        # output: (2000 / 1_000_000) * 0.60 = 0.0012
        # total: 0.0027
        assert cost == pytest.approx(0.0027)

    def test_gemini_2_0_flash_returns_correct_cost(self) -> None:
        """Gemini 2.0 Flash: legacy pricing."""
        cost = calculate_cost("gemini-2.0-flash", prompt_tokens=5_000, completion_tokens=1_000)
        assert cost is not None
        # input: (5000 / 1_000_000) * 0.10 = 0.0005
        # output: (1000 / 1_000_000) * 0.40 = 0.0004
        # total: 0.0009
        assert cost == pytest.approx(0.0009)

    def test_unknown_model_returns_none(self) -> None:
        """Unknown models should return None (no pricing data)."""
        cost = calculate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
        assert cost is None

    def test_empty_model_returns_none(self) -> None:
        """Empty model string should return None."""
        cost = calculate_cost("", prompt_tokens=1000, completion_tokens=500)
        assert cost is None

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens should return 0 cost for known models."""
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens=0, completion_tokens=0)
        assert cost is not None
        assert cost == 0.0

    def test_only_prompt_tokens(self) -> None:
        """Only prompt tokens, no completion."""
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost is not None
        assert cost == pytest.approx(1.25)

    def test_only_completion_tokens(self) -> None:
        """Only completion tokens, no prompt."""
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost is not None
        assert cost == pytest.approx(10.00)

    def test_large_token_counts(self) -> None:
        """Large realistic token counts."""
        cost = calculate_cost("gemini-2.5-pro", prompt_tokens=100_000, completion_tokens=50_000)
        assert cost is not None
        # input: (100000 / 1_000_000) * 1.25 = 0.125
        # output: (50000 / 1_000_000) * 10.00 = 0.5
        # total: 0.625
        assert cost == pytest.approx(0.625)


class TestFormatCost:
    """Tests for format_cost()."""

    def test_none_returns_na(self) -> None:
        """None cost should return 'N/A'."""
        assert format_cost(None) == "N/A"

    def test_small_cost_uses_four_decimals(self) -> None:
        """Costs under $0.01 should use 4 decimal places."""
        assert format_cost(0.0023) == "$0.0023"
        assert format_cost(0.0001) == "$0.0001"
        assert format_cost(0.009) == "$0.0090"

    def test_large_cost_uses_two_decimals(self) -> None:
        """Costs >= $0.01 should use 2 decimal places."""
        assert format_cost(0.01) == "$0.01"
        assert format_cost(0.50) == "$0.50"
        assert format_cost(1.25) == "$1.25"
        assert format_cost(10.00) == "$10.00"

    def test_zero_cost(self) -> None:
        """Zero cost should format as small cost (4 decimals)."""
        assert format_cost(0.0) == "$0.0000"

    def test_boundary_cost(self) -> None:
        """Cost exactly at the boundary ($0.01)."""
        assert format_cost(0.01) == "$0.01"
        assert format_cost(0.0099) == "$0.0099"


class TestModelPricing:
    """Tests for the ModelPricing dataclass and MODEL_PRICING dict."""

    def test_model_pricing_is_frozen(self) -> None:
        """ModelPricing should be immutable."""
        pricing = ModelPricing(input_per_1m=1.0, output_per_1m=2.0)
        with pytest.raises(AttributeError):
            pricing.input_per_1m = 5.0  # type: ignore[misc]

    def test_all_models_have_positive_pricing(self) -> None:
        """All models in MODEL_PRICING should have positive pricing."""
        for model_id, pricing in MODEL_PRICING.items():
            assert pricing.input_per_1m > 0, f"{model_id} has non-positive input pricing"
            assert pricing.output_per_1m > 0, f"{model_id} has non-positive output pricing"

    def test_known_models_exist(self) -> None:
        """Verify key models are present in the pricing dict."""
        expected_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
        for model_id in expected_models:
            assert model_id in MODEL_PRICING, f"Missing pricing for {model_id}"


class TestThinkingTokenPricing:
    """Tests for thinking token support in calculate_cost()."""

    def test_thinking_tokens_use_thinking_rate(self) -> None:
        """When thinking_per_1m is set, thinking tokens should use that rate."""
        cost = calculate_cost(
            "gemini-2.5-pro",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=1_000_000,
        )
        assert cost is not None
        # gemini-2.5-pro: thinking_per_1m = 10.00
        assert cost == pytest.approx(10.00)

    def test_thinking_tokens_combined_with_io(self) -> None:
        """Thinking tokens add to the total cost alongside prompt/completion."""
        cost = calculate_cost(
            "gemini-2.5-pro",
            prompt_tokens=1000,
            completion_tokens=500,
            thinking_tokens=2000,
        )
        assert cost is not None
        # input:    (1000 / 1M) * 1.25  = 0.00125
        # output:   (500  / 1M) * 10.00 = 0.005
        # thinking: (2000 / 1M) * 10.00 = 0.02
        expected = 0.00125 + 0.005 + 0.02
        assert cost == pytest.approx(expected)

    def test_thinking_tokens_fallback_to_output_rate(self) -> None:
        """When thinking_per_1m is None, thinking tokens use output rate."""
        # gemini-2.0-flash has no thinking_per_1m
        pricing = MODEL_PRICING.get("gemini-2.0-flash")
        assert pricing is not None
        assert pricing.thinking_per_1m is None

        cost = calculate_cost(
            "gemini-2.0-flash",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=1_000_000,
        )
        assert cost is not None
        # Should fallback to output_per_1m = 0.40
        assert cost == pytest.approx(0.40)

    def test_zero_thinking_tokens_no_extra_cost(self) -> None:
        """Zero thinking tokens should not add any cost."""
        cost_without = calculate_cost("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500, thinking_tokens=0)
        cost_baseline = calculate_cost("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        assert cost_without == cost_baseline

    def test_flash_thinking_rate(self) -> None:
        """Gemini 2.5 Flash has its own thinking rate."""
        pricing = MODEL_PRICING.get("gemini-2.5-flash")
        assert pricing is not None
        assert pricing.thinking_per_1m is not None

        cost = calculate_cost(
            "gemini-2.5-flash",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=1_000_000,
        )
        assert cost is not None
        assert cost == pytest.approx(pricing.thinking_per_1m)

    def test_thinking_per_1m_is_optional_field(self) -> None:
        """ModelPricing thinking_per_1m defaults to None."""
        pricing = ModelPricing(input_per_1m=1.0, output_per_1m=2.0)
        assert pricing.thinking_per_1m is None

    def test_models_with_thinking_have_positive_rates(self) -> None:
        """All models with thinking_per_1m set should have positive rates."""
        for model_id, pricing in MODEL_PRICING.items():
            if pricing.thinking_per_1m is not None:
                assert pricing.thinking_per_1m > 0, f"{model_id} has non-positive thinking rate"
